# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import math
from dataclasses import dataclass, field
from functools import reduce, wraps
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on
from cuda import cudart

from .._ipc_utils import IpcMemory, set_peer_access
from .._utils import pad_vocab_size, str_dtype_to_torch, trt_dtype_to_torch
from ..logger import logger
from ..mapping import Mapping
from ..quantization import QuantMode
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .lora_manager import LoraManager
from .session import _scoped_stream


def to_word_list_format(word_dict: List[List[str]],
                        tokenizer=None,
                        add_special_tokens=False):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer != None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=add_special_tokens)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def _prepare_input_ids(tensors: Sequence[torch.Tensor]):
    tensors = [torch.flatten(t) for t in tensors]
    data = torch.unsqueeze(torch.concat(tensors), 0)
    row_lengths = [t.size(0) for t in tensors]
    row_lengths = torch.tensor(row_lengths,
                               dtype=torch.int32,
                               device=data.device)
    return (data, row_lengths)


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


def _update_cuda_graph_instance(instance, graph):
    err = cudart.cudaGraphExecUpdate(instance, graph)
    if err != cudart.cudaError_t.cudaSuccess:
        # When updating cuda graph failed, destroy and instantiate one.
        CUASSERT(cudart.cudaGraphExecDestroy(instance))
        instance = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))[0]
    return instance


def _prepare_attention_mask(input_ids: torch.Tensor,
                            pad_id: Optional[int] = None):
    is_pad_id_in_inputs = (pad_id is not None) and (pad_id in input_ids)
    if input_ids is not None and is_pad_id_in_inputs:
        return input_ids.ne(pad_id).int()
    else:
        return torch.ones(input_ids.shape,
                          dtype=torch.int32,
                          device=input_ids.device)


def _tile_beam_width(tensor: torch.Tensor, num_beams: int):
    new_shape = np.array(tensor.shape)
    new_shape[0] = new_shape[0] * num_beams

    tile_size = np.ones(new_shape.shape, dtype=np.int32)
    tile_size = np.insert(tile_size, 1, num_beams)

    new_tensor = torch.unsqueeze(tensor, 1)
    new_tensor = new_tensor.tile(tile_size.tolist())
    new_tensor = new_tensor.reshape(new_shape.tolist())
    return new_tensor


class _Runtime(object):
    runtime_rank: int
    runtime: trt.Runtime
    engine: trt.ICudaEngine
    ctx_context: trt.IExecutionContext
    context_0: trt.IExecutionContext
    context_1: trt.IExecutionContext
    cuda_graph_instances: List[cudart.cudaGraphExec_t]

    def __init__(self, engine_buffer, mapping: Mapping):
        self.__prepare(mapping, engine_buffer)

    def __create_and_setup_context(self, address, profile_idx,
                                   stream) -> trt.IExecutionContext:
        context = self.engine.create_execution_context_without_device_memory()
        assert context is not None
        context.device_memory = address
        context.set_optimization_profile_async(profile_idx, stream)
        return context

    def __prepare(self, mapping: Mapping, engine_buffer):
        self.runtime_rank = mapping.rank
        local_rank = self.runtime_rank % mapping.gpus_per_node
        torch.cuda.set_device(local_rank)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.runtime = trt.Runtime(logger.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        assert self.engine is not None
        # The device_memory_size stores the memory required by the largest profile
        address = CUASSERT(cudart.cudaMalloc(self.engine.device_memory_size))[0]
        self.address = address

        # cuda graph ping-pong instances
        self.cuda_graph_instances = [None for _ in range(2)]

        with _scoped_stream() as stream:
            if self.engine.num_optimization_profiles == 1:
                # At step = 0, context_1 is active
                # At step = 1, context_0 is active
                # At step = 2, context_1 is active
                self.context_0 = self.__create_and_setup_context(
                    address, 0, stream)
                self.context_1 = self.__create_and_setup_context(
                    address, 0, stream)
                self.ctx_context = self.context_1
            elif self.engine.num_optimization_profiles == 2:
                # At step = 0, ctx_context is active
                # At step = 1, context_0 is active
                # At step = 2, context_1 is active
                self.ctx_context = self.__create_and_setup_context(
                    address, 0, stream)
                self.context_0 = self.__create_and_setup_context(
                    address, 1, stream)
                self.context_1 = self.__create_and_setup_context(
                    address, 1, stream)
            else:
                assert False, "Maximum of up to two optimization profiles only"

    def _set_shape(self, context: trt.IExecutionContext,
                   shape_dict: Dict[str, List[int]]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if not name in shape_dict:
                # shape and buffer can be set by calling _set_tensors API
                continue
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                ok = context.set_input_shape(name, shape_dict[name])
                logger.debug(
                    f"setting input tensor {name} with shape {shape_dict[name]}"
                )
                if not ok:
                    raise ValueError(
                        f"Couldn't assign {name} with shape {shape_dict[name]}, "
                        f"engine supports [min, opt, max] = {self.engine.get_tensor_profile_shape(name, self.engine.active_optimization_profile)}"
                    )

    def _set_buffer(self, context: trt.IExecutionContext,
                    buffer_dict: Dict[str, torch.Tensor]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in buffer_dict.keys():
                dtype = self.engine.get_tensor_dtype(name)
                shape = context.get_tensor_shape(name)
                buffer_dict[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device='cuda')
            assert buffer_dict[name].is_contiguous(
            ), f"{name} is not contiguous()"
            context.set_tensor_address(name, buffer_dict[name].data_ptr())

    def _set_tensors(self, context: trt.IExecutionContext,
                     tensors: Dict[str, "RuntimeTensor"]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # it's allowed to call set_tensors multi times with different tensors
            # each time only set some of the engine tensors, so it is valid to skip the ones not in the current given tensors dict
            if not name in tensors:
                if self.engine.get_tensor_mode(
                        t.name) == trt.TensorIOMode.OUTPUT:
                    dtype = self.engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    tensors[name] = RuntimeTensor.from_torch(
                        name,
                        torch.zeros(tuple(shape),
                                    dtype=trt_dtype_to_torch(dtype),
                                    device='cuda'))
                else:
                    continue
            t = tensors[name]
            # output's shape is inference by TRT, no need to set the shape here
            if self.engine.get_tensor_mode(t.name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(t.name, t.shape)
            context.set_tensor_address(t.name, t.data)

    def _run(self,
             context: trt.IExecutionContext,
             stream: Union[int, torch.cuda.Stream] = None) -> bool:
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        elif isinstance(stream, torch.cuda.Stream):
            stream = stream.cuda_stream
        ok = context.execute_async_v3(stream)
        return ok

    def __del__(self):
        cudart.cudaFree(self.address)


@dataclass
class ModelConfig:
    vocab_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    gpt_attention_plugin: bool
    remove_input_padding: bool = False
    model_name: str = ""
    paged_kv_cache: bool = False
    cross_attention: bool = False
    head_size: int = None
    has_position_embedding: bool = True
    has_token_type_embedding: bool = False
    tokens_per_block: int = 64
    max_prompt_embedding_table_size: int = 0
    quant_mode: QuantMode = QuantMode(0)
    gather_all_token_logits: bool = False
    dtype: str = ""
    use_custom_all_reduce: bool = False
    lora_plugin: bool = False
    lora_target_modules: List[str] = field(default_factory=list)


@dataclass
class SamplingConfig:
    end_id: int
    pad_id: int

    max_new_tokens: int = field(default=20)
    num_beams: int = field(default=1)
    max_attention_window_size: Optional[int] = field(default=None)
    output_sequence_lengths: bool = field(default=False)
    return_dict: bool = field(default=False)
    stop_words_list: Optional[torch.Tensor] = field(default=None)
    bad_words_list: Optional[torch.Tensor] = field(default=None)

    temperature: Union[float, torch.Tensor] = field(default=1.0)
    top_k: Union[int, torch.Tensor] = field(default=1)
    top_p: Union[float, torch.Tensor] = field(default=0.0)
    top_p_decay: Optional[float] = field(default=None)
    top_p_min: Optional[float] = field(default=None)
    top_p_reset_ids: Optional[int] = field(default=None)

    length_penalty: Union[float, torch.Tensor] = field(default=1.0)
    repetition_penalty: Union[float, torch.Tensor] = field(default=1.0)
    min_length: Union[int, torch.Tensor] = field(default=1)
    presence_penalty: Union[float, torch.Tensor] = field(default=0.0)
    use_beam_hyps: bool = field(default=True)

    ## None here means user didn't set it, and dynamicDecodeOp.cpp take optional value
    ## The real default value is set in dynamicDecodeOp.cpp when it's None
    beam_search_diversity_rate: Union[float, torch.Tensor] = field(init=False,
                                                                   default=0.0)
    random_seed: Union[int, torch.Tensor] = field(init=False, default=None)
    output_cum_log_probs: bool = field(init=False, default=False)
    output_log_probs: bool = field(init=False, default=False)

    def update(self, **kwargs):
        unused_kwargs = dict()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                unused_kwargs[key] = value
        return unused_kwargs


class LogitsProcessor:
    """
    Base class for all logit processors that can be applied during generation.
    """

    def __call__(self, step: int, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list, LogitsProcessor):

    def __call__(self, step: int, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> torch.Tensor:
        for processor in self:
            scores = processor(step, input_ids, scores)
        return scores


class StoppingCriteria:
    """
    Base class for all stopping criteria that can be applied during generation.
    """

    def __call__(self, step: int, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class StoppingCriteriaList(list, StoppingCriteria):

    def __call__(self, step: int, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> bool:
        return any(criteria(step, input_ids, scores) for criteria in self)


class RuntimeTensor:

    def __init__(self):
        self._name = ""
        # shape is the one sent to TRT, the actual torch tensor can be larger than the shape
        # this is useful when allocating a big KV cache tensor at the beginning and incremental seq length dim of TRT engine's input tensor
        self._shape = None
        self._torch_tensor = None

    @staticmethod
    def from_torch(
            name: str,
            data: torch.Tensor,
            override_shape: Optional[Iterable] = None) -> 'RuntimeTensor':
        assert (isinstance(data, torch.Tensor))
        t = RuntimeTensor()
        t._name = name
        # need to hold the torch tensor for memory life time
        t._torch_tensor = data.contiguous()
        torch_shape = list(data.size())
        if override_shape is not None:
            t._shape = override_shape
            assert isinstance(override_shape, list) or isinstance(
                override_shape, tuple)
            assert all([lambda x: x >= 0 for x in override_shape
                        ]), f"Expect all dimensions >=0, got {override_shape}"
            volumeFunc = lambda dims: reduce(lambda x, y: x * y, dims, 1)
            assert volumeFunc(override_shape) <= volumeFunc(torch_shape), \
                f"Override the shape to be larger than the underlying torch Tensor, got {override_shape}, torch tensor shape {torch_shape}"
        else:
            t._shape = torch_shape
        return t

    def to_torch(self) -> torch.Tensor:
        return self._torch_tensor

    @property
    def shape(self) -> Iterable[int]:
        return self._shape

    @property
    def data(self):
        return self._torch_tensor.data_ptr()

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> torch.dtype:
        return self._torch_tensor.dtype


class GenerationSession(object):

    _model_config: ModelConfig
    mapping: Mapping
    runtime: _Runtime
    device: torch.device
    batch_size: int
    buffer_allocated: bool
    debug_mode: bool
    quant_mode: QuantMode
    cuda_graph_mode: bool
    dtype: trt.DataType
    debug_tensors_to_save: None

    def __init__(self,
                 model_config: ModelConfig,
                 engine_buffer,
                 mapping: Mapping,
                 debug_mode=False,
                 debug_tensors_to_save=None,
                 cuda_graph_mode=False,
                 stream: torch.cuda.Stream = None):
        assert isinstance(model_config, ModelConfig)
        self._model_config = model_config
        self.mapping = mapping
        self.runtime = _Runtime(engine_buffer, mapping)
        self.device = torch.device(
            f'cuda:{self.runtime.runtime_rank % mapping.gpus_per_node}')
        torch.cuda.set_device(self.device)
        # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)
        self.debug_mode = debug_mode
        self.debug_tensors_to_save = debug_tensors_to_save

        self.cuda_graph_mode = cuda_graph_mode
        # Optional inputs for dynamic decoder
        self.top_p_decay = None
        self.top_p_min = None
        self.top_p_reset_ids = None
        #TODO: in tensorrt_llm/cpp/tensorrt_llm/thop/dynamicDecodeOp.cpp it's T, can be float or half?
        self.embedding_bias_opt = None

        self.buffer = None
        self.buffer_allocated = False

        self.vocab_size_padded = pad_vocab_size(self.vocab_size,
                                                self.mapping.tp_size)

        if self.paged_kv_cache:
            logger.warning(
                "The paged KV cache in Python runtime is experimental. For performance and correctness, please, use C++ runtime."
            )

        # if self.mapping.has_pp():
        #     self.nccl_comm = torch.classes.FasterTransformer.NcclCommunicatorOp(
        #         self.mapping.tp_size, self.mapping.pp_size, self.mapping.rank)

        if self.mapping.is_last_pp_rank():
            self.decoder_logits_dtype = self._tensor_dtype('logits')
            if self.decoder_logits_dtype not in [torch.float16, torch.float32]:
                logger.warning(
                    "Logits dtype not supported by decoder. Falling back to float32. You may want to change the logits dtype to float16 in your model definition."
                )
                self.decoder_logits_dtype = torch.float32
            self.dynamic_decoder = torch.classes.FasterTransformer.DynamicDecodeOp(
                self.vocab_size, self.vocab_size_padded, self.mapping.tp_size,
                self.mapping.pp_size, self.decoder_logits_dtype)

        self.gather_tree = torch.ops.tensorrt_llm.gather_tree

        expected_tensor_names = []
        if self.mapping.is_first_pp_rank():
            expected_tensor_names += ['input_ids']
        else:
            expected_tensor_names += ['hidden_states_input']

        if self.mapping.is_last_pp_rank():
            expected_tensor_names += ['logits']
            if not model_config.gather_all_token_logits:
                expected_tensor_names += ['last_token_ids']
        else:
            expected_tensor_names += ['hidden_states_output']

        if model_config.has_position_embedding and self.mapping.is_first_pp_rank(
        ):
            expected_tensor_names += ['position_ids']
        if model_config.has_token_type_embedding and self.mapping.is_first_pp_rank(
        ):
            expected_tensor_names += ['token_type_ids']

        expected_tensor_names += ['cache_indirection']

        if self.paged_kv_cache:
            expected_tensor_names += [
                f'kv_cache_block_pointers_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
            expected_tensor_names += [
                f'host_kv_cache_block_pointers_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
        else:
            expected_tensor_names += [
                f'past_key_value_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
            expected_tensor_names += [
                f'present_key_value_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]

        if model_config.gpt_attention_plugin:
            expected_tensor_names += [
                'sequence_length', 'context_lengths', 'host_request_types',
                'host_past_key_value_lengths'
            ]
            expected_tensor_names += [
                f'host_max_attention_window_size_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
            if model_config.remove_input_padding:
                expected_tensor_names.append('host_context_lengths')
        else:
            expected_tensor_names += [
                'attention_mask',
            ]

        if model_config.max_prompt_embedding_table_size > 0:
            expected_tensor_names += [
                'prompt_embedding_table', 'tasks', 'prompt_vocab_size'
            ]

        if model_config.cross_attention:
            expected_tensor_names += [
                f'cross_present_key_value_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
            expected_tensor_names += [
                f'cross_past_key_value_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
            expected_tensor_names += [
                'encoder_output', 'encoder_input_lengths',
                'encoder_max_input_length'
            ]

        if self.mapping.tp_size > 1 and model_config.use_custom_all_reduce:
            expected_tensor_names += ['all_reduce_workspace']

        self.lora_target_modules = model_config.lora_target_modules

        if model_config.lora_plugin:
            for lora_module in self.lora_target_modules:
                expected_tensor_names += [
                    f'{lora_module}_lora_ranks_{i}'
                    for i in range(self.first_layer, self.last_layer)
                ]

                expected_tensor_names += [
                    f'{lora_module}_lora_weights_pointers_{i}'
                    for i in range(self.first_layer, self.last_layer)
                ]

        found_tensor_names = [
            self.runtime.engine.get_tensor_name(i)
            for i in range(self.runtime.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected, to use this GenerationSession, " \
                    "you need to use GPTLMHeadModel.prepare_inputs to create TRT Network inputs."
            )
        if self.debug_mode:
            self.debug_tensors = list(
                set(found_tensor_names) - set(expected_tensor_names))

    @property
    def vocab_size(self):
        return self._model_config.vocab_size

    @property
    def num_layers(self):
        assert self._model_config.num_layers % self.mapping.pp_size == 0, \
            f"num_layers {self._model_config.num_layers} must be a multiple of pipeline parallelism size {self.mapping.pp_size}"
        return self._model_config.num_layers // self.mapping.pp_size

    @property
    def first_layer(self):
        return self.num_layers * self.mapping.pp_rank

    @property
    def last_layer(self):
        return self.first_layer + self.num_layers

    @property
    def num_heads(self):
        return self._model_config.num_heads

    @property
    def hidden_size(self):
        return self._model_config.hidden_size

    @property
    def use_gpt_attention_plugin(self):
        return self._model_config.gpt_attention_plugin

    @property
    def paged_kv_cache(self):
        return self._model_config.paged_kv_cache

    @property
    def tokens_per_block(self):
        return self._model_config.tokens_per_block

    @property
    def remove_input_padding(self):
        return self._model_config.remove_input_padding

    @property
    def num_heads_kv(self):
        return self._model_config.num_kv_heads

    @property
    def head_size(self):
        return self.hidden_size // self.num_heads if self._model_config.head_size is None else self._model_config.head_size

    @property
    def max_prompt_embedding_table_size(self):
        return self._model_config.max_prompt_embedding_table_size

    @property
    def quant_mode(self):
        return self._model_config.quant_mode

    @property
    def gather_all_token_logits(self):
        return self._model_config.gather_all_token_logits

    @property
    def dtype(self):
        return str_dtype_to_torch(self._model_config.dtype)

    @property
    def use_custom_all_reduce(self):
        return self._model_config.use_custom_all_reduce

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    @property
    def cross_attention(self):
        return self._model_config.cross_attention

    @property
    def has_position_embedding(self):
        return self._model_config.has_position_embedding

    @property
    def has_token_type_embedding(self):
        return self._model_config.has_token_type_embedding

    @property
    def use_lora_plugin(self):
        return self._model_config.lora_plugin

    def __setup_decoder(self, input_ids: torch.Tensor,
                        sampling_config: SamplingConfig,
                        host_context_lengths: torch.Tensor):
        '''Allocate buffers and setup the post-processing decoder kernel
        '''
        batch_size = host_context_lengths.shape[0]
        scfg = sampling_config  # just to make a shorter name, no other meaning
        if isinstance(scfg.top_k, torch.Tensor):
            assert scfg.top_k.dtype == torch.int32, f"scfg.top_k.dtype ({scfg.top_k.dtype}) must be torch.int32"
            assert scfg.top_k.shape[
                0] == batch_size, f"scfg.top_k.shape[0] ({scfg.top_k.shape[0]}) must equal to batch_size ({batch_size})"
            self.top_k = scfg.top_k
        else:
            self.top_k = torch.full([batch_size], scfg.top_k, dtype=torch.int32)

        if isinstance(scfg.top_p, torch.Tensor):
            assert scfg.top_p.dtype == torch.float32, f"scfg.top_p.dtype ({scfg.top_p.dtype}) must be torch.float32"
            assert scfg.top_p.shape[
                0] == batch_size, f"scfg.top_p.shape[0] ({scfg.top_p.shape[0]}) must equal to batch_size ({batch_size})"
            self.top_p = scfg.top_p
        else:
            self.top_p = torch.full([batch_size],
                                    scfg.top_p,
                                    dtype=torch.float32)

        if isinstance(scfg.temperature, torch.Tensor):
            assert scfg.temperature.dtype == torch.float32, f"scfg.temperature.dtype ({scfg.temperature.dtype}) must be torch.float32"
            assert scfg.temperature.shape[
                0] == batch_size, f"scfg.temperature.shape[0] ({scfg.temperature.shape[0]}) must equal to batch_size ({batch_size})"
            self.temperature = scfg.temperature
        else:
            self.temperature = torch.full([batch_size],
                                          scfg.temperature,
                                          dtype=torch.float32)

        if isinstance(scfg.repetition_penalty, torch.Tensor):
            assert scfg.repetition_penalty.dtype == torch.float32, f"scfg.repetition_penalty.dtype ({scfg.repetition_penalty.dtype}) must be torch.float32"
            assert scfg.repetition_penalty.shape[
                0] == batch_size, f"scfg.repetition_penalty.shape[0] ({scfg.repetition_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.repetition_penalty = scfg.repetition_penalty
        elif scfg.repetition_penalty == 1.0:
            self.repetition_penalty = None
        else:
            self.repetition_penalty = torch.full([batch_size],
                                                 scfg.repetition_penalty,
                                                 dtype=torch.float32)

        self.host_length_penalty = torch.full([batch_size],
                                              scfg.length_penalty,
                                              dtype=torch.float32)
        self.length_penalty = self.host_length_penalty.to(self.device)

        if isinstance(scfg.presence_penalty, torch.Tensor):
            assert scfg.presence_penalty.dtype == torch.float32, f"scfg.presence_penalty.dtype ({scfg.presence_penalty.dtype}) must be torch.float32"
            assert scfg.presence_penalty.shape[
                0] == batch_size, f"scfg.presence_penalty.shape[0] ({scfg.presence_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.presence_penalty = scfg.presence_penalty
        elif scfg.presence_penalty == 0.0:
            self.presence_penalty = None
        else:
            self.presence_penalty = torch.full([batch_size],
                                               scfg.presence_penalty,
                                               dtype=torch.float32)

        assert (
            scfg.presence_penalty == 0.0 or scfg.repetition_penalty == 1.0
        ), f"presence_penalty({scfg.presence_penalty}) and repetition_penalty({scfg.repetition_penalty}) cannot be non-default values at the same time."

        if isinstance(scfg.min_length, torch.Tensor):
            assert scfg.min_length.dtype == torch.int32, f"scfg.min_length.dtype ({scfg.min_length.dtype}) must be torch.int32"
            assert scfg.min_length.shape[
                0] == batch_size, f"scfg.min_length.shape[0] ({scfg.min_length.shape[0]}) must equal to batch_size ({batch_size})"
            self.min_length = scfg.min_length
        else:
            self.min_length = torch.full([batch_size],
                                         scfg.min_length,
                                         dtype=torch.int32)

        if isinstance(scfg.beam_search_diversity_rate, torch.Tensor):
            assert scfg.beam_search_diversity_rate.dtype == torch.float32, f"scfg.beam_search_diversity_rate.dtype ({scfg.beam_search_diversity_rate.dtype}) must be torch.float32"
            assert scfg.beam_search_diversity_rate.shape[
                0] == batch_size, f"scfg.beam_search_diversity_rate.shape[0] ({scfg.beam_search_diversity_rate.shape[0]}) must equal to batch_size ({batch_size})"
            self.beam_search_diversity_rate = scfg.beam_search_diversity_rate
        elif scfg.beam_search_diversity_rate is not None:
            self.beam_search_diversity_rate = torch.full(
                [batch_size],
                scfg.beam_search_diversity_rate,
                dtype=torch.float32)
        else:
            self.beam_search_diversity_rate = None

        if isinstance(scfg.random_seed, torch.Tensor):
            assert scfg.random_seed.dtype == torch.int64, f"scfg.random_seed.dtype ({scfg.random_seed.dtype}) must be torch.int64"
            assert scfg.random_seed.shape[
                0] == batch_size, f"scfg.random_seed.shape[0] ({scfg.random_seed.shape[0]}) must equal to batch_size ({batch_size})"
            self.random_seed = scfg.random_seed
        elif scfg.random_seed is not None:
            self.random_seed = torch.full([batch_size],
                                          scfg.random_seed,
                                          dtype=torch.int64)
        else:
            self.random_seed = None

        if self.mapping.is_last_pp_rank():
            self.dynamic_decoder.setup(batch_size, scfg.num_beams, self.top_k,
                                       self.top_p, self.temperature,
                                       self.repetition_penalty,
                                       self.presence_penalty, self.min_length,
                                       self.host_length_penalty,
                                       self.beam_search_diversity_rate,
                                       self.random_seed, self.top_p_decay,
                                       self.top_p_min, self.top_p_reset_ids)

        assert scfg.end_id is not None, "end_id cannot be none"
        assert scfg.pad_id is not None, 'pad_id cannot be none'
        self.end_ids = torch.full((batch_size * scfg.num_beams, ),
                                  scfg.end_id,
                                  dtype=torch.int32,
                                  device=self.device)
        max_context_length = host_context_lengths.max()

        # setup output ids buffer
        if input_ids.shape[0] != host_context_lengths.shape[0]:
            # dim 0 of input_ids is not batch size, which means remove_padding is enabled
            split_ids_list = list(
                torch.split(input_ids,
                            host_context_lengths.numpy().tolist(),
                            dim=1))
            padded_input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(split_ids_list,
                                           dtype=torch.int32,
                                           device='cuda'),
                scfg.pad_id).reshape(batch_size, max_context_length)
        else:
            padded_input_ids = input_ids
        if scfg.num_beams > 1:
            tiled_input_ids = _tile_beam_width(padded_input_ids, scfg.num_beams)
            tiled_input_ids = tiled_input_ids.reshape(batch_size,
                                                      scfg.num_beams,
                                                      max_context_length)
            tiled_input_ids.permute(2, 0, 1)  # TODO: delete?
            self.output_ids = torch.cat(
                (tiled_input_ids,
                 torch.full((batch_size, scfg.num_beams,
                             self.max_seq_length - max_context_length),
                            scfg.end_id,
                            dtype=padded_input_ids.dtype,
                            device=padded_input_ids.device)),
                axis=-1)
        else:
            self.output_ids = torch.cat(
                (padded_input_ids,
                 torch.full(
                     (batch_size, self.max_seq_length - max_context_length),
                     scfg.end_id,
                     dtype=padded_input_ids.dtype,
                     device=padded_input_ids.device)),
                axis=-1)

        # Note: we still allocate max_seq_length size of parent ids (not max_attention_window_size).
        self.parent_ids = torch.zeros(
            (batch_size, scfg.num_beams, self.max_seq_length),
            dtype=torch.int32,
            device=self.device)

        if scfg.num_beams > 1:
            self.new_tokens = torch.zeros([batch_size, scfg.num_beams, 1],
                                          dtype=torch.int32,
                                          device=self.device)
        else:
            self.new_tokens = torch.zeros([batch_size, 1],
                                          dtype=torch.int32,
                                          device=self.device)

        if scfg.num_beams > 1 or scfg.output_cum_log_probs:
            self.cum_log_probs = torch.full((batch_size, scfg.num_beams),
                                            -1e20,
                                            dtype=torch.float32,
                                            device=self.device)
            self.cum_log_probs[:, 0] = 0.0
        else:
            self.cum_log_probs = None

        if scfg.output_log_probs:
            self.log_probs = torch.zeros(
                (self.max_new_tokens, batch_size, scfg.num_beams),
                dtype=torch.float32,
                device=self.device)
        else:
            self.log_probs = None

        self.finished = torch.zeros((batch_size, scfg.num_beams),
                                    dtype=torch.uint8,
                                    device=self.device)

        if scfg.use_beam_hyps:
            self.beam_hyps_output_ids_tgt = torch.full(
                size=[batch_size, scfg.num_beams * 2, self.max_seq_length],
                fill_value=scfg.end_id,
                dtype=torch.int32,
                device=self.device)
            self.beam_hyps_sequence_lengths_tgt = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.int32,
                device=self.device)
            self.beam_hyps_cum_log_probs = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_normed_scores = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_log_probs = torch.zeros(
                [batch_size, scfg.num_beams * 2, self.max_seq_length],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_min_normed_scores = torch.zeros([batch_size],
                                                           dtype=torch.float,
                                                           device=self.device)
            self.beam_hyps_num_beams = torch.zeros([batch_size],
                                                   dtype=torch.int32,
                                                   device=self.device)
            self.beam_hyps_is_done = torch.zeros([batch_size],
                                                 dtype=torch.bool,
                                                 device=self.device)
        else:
            self.beam_hyps_output_ids_tgt = None
            self.beam_hyps_sequence_lengths_tgt = None
            self.beam_hyps_cum_log_probs = None
            self.beam_hyps_normed_scores = None
            self.beam_hyps_log_probs = None
            self.beam_hyps_min_normed_scores = None
            self.beam_hyps_num_beams = None
            self.beam_hyps_is_done = None

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.runtime.engine.get_tensor_dtype(name))
        return dtype

    def setup(self,
              batch_size: int,
              max_context_length: int,
              max_new_tokens: int,
              beam_width: int = 1,
              max_attention_window_size: Optional[int] = None,
              encoder_max_input_length: Optional[int] = None,
              lora_manager: LoraManager = None,
              lora_uids: List[str] = None):
        # Store these params related to buffer size to check against
        # the input shape with the params given in decode()
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_context_length + max_new_tokens
        self.beam_width = beam_width
        self.encoder_max_input_length = encoder_max_input_length
        if max_attention_window_size is None:
            self.max_attention_window_size = self.max_seq_length
            logger.debug(
                "The max_attention_window_size is not set, we will use max_seq_length by default."
            )
            self.host_max_attention_window_sizes = [
                torch.ones(
                    (1, ), dtype=torch.int32) * self.max_attention_window_size
                for i in range(self.num_layers)
            ]
        elif isinstance(max_attention_window_size, int):
            if max_attention_window_size > self.max_seq_length:
                logger.warning(
                    "The value of max_attention_window_size should ideally not exceed max_seq_length. "
                    "Therefore, it has been adjusted to match the value of max_seq_length."
                )
            self.max_attention_window_size = min(max_attention_window_size,
                                                 self.max_seq_length)
            self.host_max_attention_window_sizes = [
                torch.ones(
                    (1, ), dtype=torch.int32) * self.max_attention_window_size
                for i in range(self.num_layers)
            ]
        elif isinstance(max_attention_window_size, torch.Tensor):
            self.max_attention_window_size = int(
                torch.max(max_attention_window_size).item())
            if self.max_attention_window_size > self.max_seq_length:
                logger.warning(
                    "The value of max_attention_window_size should ideally not exceed max_seq_length. "
                    "Therefore, it has been adjusted to match the value of max_seq_length."
                )
            self.max_attention_window_size = min(self.max_attention_window_size,
                                                 self.max_seq_length)
            if max_attention_window_size.shape[0] != self.num_layers:
                logger.error(
                    "max_attention_window_size tensor's size is not equal to num_layers! "
                    "Note that num_layers = num_total_layers // pipeline_parallelism_size."
                )
                assert False
            self.host_max_attention_window_sizes = [
                torch.minimum(
                    max_attention_window_size.to(torch.int32)[i],
                    torch.IntTensor([self.max_seq_length]))
                for i in range(self.num_layers)
            ]
        else:
            assert False, "invalid max_attention_window_size!"
        self.lora_manager = lora_manager

        self.buffer = {}
        if self.mapping.is_last_pp_rank():
            self.buffer['logits'] = torch.empty(
                (batch_size, self.vocab_size_padded)
                if not self.gather_all_token_logits else
                (batch_size, max_context_length, self.vocab_size_padded),
                dtype=self._tensor_dtype('logits'),
                device=self.device)
        if self.cross_attention:
            # use shape info to pass max length info in remove padding mode
            self.buffer['encoder_max_input_length'] = torch.empty(
                (encoder_max_input_length, ),
                dtype=self._tensor_dtype('encoder_max_input_length'),
                device=self.device)

        if self.paged_kv_cache:
            blocks = batch_size * beam_width * math.ceil(
                self.max_attention_window_size / self.tokens_per_block)
            cache_shape = (
                blocks,
                2,
                self.num_heads_kv,
                self.tokens_per_block,
                self.head_size,
            )
        else:
            cache_shape = (
                batch_size,
                2,
                self.num_heads_kv,
                self.max_attention_window_size,
                self.head_size,
            )
            if self.cross_attention:
                cross_cache_shape = (
                    batch_size,
                    2,
                    self.num_heads_kv,
                    self.encoder_max_input_length,
                    self.head_size,
                )

        for i in range(self.first_layer, self.last_layer):
            if self.quant_mode.has_kv_cache_quant():
                # Since torch does not support fp8 now, using int8 here.
                kv_cache_type = torch.int8
            else:
                kv_cache_type = self.dtype if self.paged_kv_cache else self._tensor_dtype(
                    f'present_key_value_{i}')
            self.buffer[f'present_key_value_{i}'] = torch.empty(
                cache_shape, dtype=kv_cache_type, device=self.device)
            if self.cross_attention:
                self.buffer[f'cross_present_key_value_{i}'] = torch.empty(
                    cross_cache_shape, dtype=kv_cache_type, device=self.device)

        if self.use_gpt_attention_plugin:
            self.sequence_length_buffer = torch.ones((batch_size, ),
                                                     dtype=torch.int32,
                                                     device=self.device)
        else:
            # without plugin, we need two set of kv cache buffers,
            # one for inputs, and the other for outputs.
            # They will take turns to act as input and output buffers.
            # Not applicable to cross KV buffers as it's constant
            for i in range(self.first_layer, self.last_layer):
                trt_dtype = self.runtime.engine.get_tensor_dtype(
                    f'present_key_value_{i}')
                if trt_dtype == trt.fp8:
                    # PyTorch doesn't support fp8 datatype, use int8 instead of it because int8 datatype size is same with fp8.
                    # TODO: Remove this section when PyTorch support fp8 datatype
                    dtype = torch.int8
                else:
                    dtype = self._tensor_dtype(f'present_key_value_{i}')
                self.buffer[f'1_present_key_value_{i}'] = torch.empty(
                    cache_shape, dtype=dtype, device=self.device)

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            set_peer_access(self.mapping)
            float_element_size = torch.tensor([],
                                              dtype=torch.float).element_size()
            buffer_size = batch_size * beam_width * max_context_length * self.hidden_size * self.mapping.tp_size * float_element_size
            barrier_size = IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * self.mapping.tp_size

            self.ipc_buffers = IpcMemory(self.mapping, buffer_size)
            self.ipc_barriers_in = IpcMemory(self.mapping, barrier_size)
            self.ipc_barriers_out = IpcMemory(self.mapping, barrier_size)
            self.all_reduce_workspace = torch.tensor(
                self.ipc_buffers.serialize() +
                self.ipc_barriers_in.serialize() +
                self.ipc_barriers_out.serialize(),
                dtype=torch.int64,
                device="cpu")

        if self.use_lora_plugin and self.lora_manager is not None:
            assert lora_uids is not None
            lora_weights_pointers_list = [
                torch.zeros(size=(batch_size, 2),
                            dtype=torch.int64).contiguous().cpu()
                for _ in range(self.num_layers)
            ]

            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer

                for lora_module in self.lora_target_modules:
                    self.buffer.update({
                        f'{lora_module}_lora_ranks_{layer_idx}':
                        torch.zeros(size=(batch_size, ),
                                    dtype=torch.int32).contiguous().cpu()
                    })

                    self.buffer.update({
                        f'{lora_module}_lora_weights_pointers_{layer_idx}':
                        torch.zeros(size=(batch_size, 2),
                                    dtype=torch.int64).contiguous().cpu()
                    })
                    for batch_idx in range(batch_size):
                        lora_uid = lora_uids[batch_idx]
                        if lora_uid is not None and lora_uid != "-1":
                            self.buffer[f'{lora_module}_lora_ranks_{layer_idx}'][
                                batch_idx] = self.lora_manager.uid_to_low_ranks(
                                    lora_uid)[layer_idx][lora_module]

                            self.buffer[
                                f'{lora_module}_lora_weights_pointers_{layer_idx}'][
                                    batch_idx][
                                        0] = self.lora_manager.lora_weights_pointers_list[
                                            layer_idx][lora_uid][lora_module][0]
                            self.buffer[
                                f'{lora_module}_lora_weights_pointers_{layer_idx}'][
                                    batch_idx][
                                        1] = self.lora_manager.lora_weights_pointers_list[
                                            layer_idx][lora_uid][lora_module][1]
                        else:
                            self.buffer[
                                f'{lora_module}_lora_ranks_{layer_idx}'][
                                    batch_idx] = 0

        self.buffer_allocated = True

    def _get_context_shape_buffer(
            self,
            input_ids: torch.Tensor,
            context_lengths: torch.Tensor,
            host_context_lengths: torch.Tensor,
            position_ids: torch.Tensor,
            last_token_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            cache_indirection: torch.Tensor,
            kv_cache_block_pointers: List[torch.Tensor],
            host_kv_cache_block_pointers: List[torch.Tensor],
            hidden_states_input: torch.Tensor = None,
            prompt_embedding_table: torch.Tensor = None,
            tasks: torch.Tensor = None,
            prompt_vocab_size: torch.Tensor = None,
            encoder_output: torch.Tensor = None,
            encoder_input_lengths: torch.Tensor = None) -> List[RuntimeTensor]:
        tensors = {}
        sym = lambda x, name: RuntimeTensor.from_torch(name, x)
        add_tensor = lambda x, name: tensors.update({name: sym(x, name)})
        add_tensor_with_shape = lambda x, name, shape: tensors.update(
            {name: RuntimeTensor.from_torch(name, x, override_shape=shape)})
        if self.use_gpt_attention_plugin:
            add_tensor(context_lengths, 'context_lengths')
        add_tensor(cache_indirection, 'cache_indirection')

        if self.has_position_embedding:
            add_tensor(position_ids, 'position_ids')

        if self.cross_attention:
            add_tensor(encoder_output, 'encoder_output')
            add_tensor(encoder_input_lengths, 'encoder_input_lengths')
            add_tensor(self.buffer['encoder_max_input_length'],
                       'encoder_max_input_length')

        if self.mapping.has_pp():
            hidden_size = self.hidden_size * self.mapping.tp_size
            hidden_states_input = hidden_states_input.resize_(
                input_ids.shape[0], input_ids.shape[1], hidden_size)

        if self.mapping.is_last_pp_rank():
            add_tensor(self.buffer['logits'], 'logits')

            if not self.gather_all_token_logits:
                add_tensor(last_token_ids, 'last_token_ids')
        else:
            add_tensor(hidden_states_input, 'hidden_states_output')

        if self.mapping.is_first_pp_rank():
            add_tensor(input_ids, 'input_ids')
        else:
            add_tensor(hidden_states_input, 'hidden_states_input')

        if prompt_embedding_table is not None:
            add_tensor(prompt_embedding_table, 'prompt_embedding_table')

            if self.remove_input_padding:
                tasks_generation = torch.concat([
                    torch.full([context_lengths[b].item()],
                               tasks[b].item(),
                               dtype=torch.int32)
                    for b in range(context_lengths.size(0))
                ]).unsqueeze(0).cuda()
            else:
                tasks_generation = tasks.unsqueeze(-1)
            add_tensor(tasks_generation, 'tasks')
            add_tensor(prompt_vocab_size, 'prompt_vocab_size')

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer
                buffer = kv_cache_block_pointers[idx].contiguous()
                shape = kv_cache_block_pointers[idx].shape
                shape = [shape[0] * shape[1], *shape[2:]]
                add_tensor_with_shape(buffer,
                                      f'kv_cache_block_pointers_{layer_idx}',
                                      shape)
                add_tensor_with_shape(
                    host_kv_cache_block_pointers[idx],
                    f'host_kv_cache_block_pointers_{layer_idx}', shape)

        batch_size = context_lengths.shape[0]
        if not self.paged_kv_cache:
            for idx in range(self.first_layer, self.last_layer):
                if not self.use_gpt_attention_plugin:
                    kv_cache_shape = (batch_size, 2, self.num_heads_kv, 0,
                                      self.head_size)
                    # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                    kv_cache_buffer = torch.zeros((1, ),
                                                  dtype=torch.float32,
                                                  device=self.device)
                    add_tensor_with_shape(kv_cache_buffer,
                                          f'past_key_value_{idx}',
                                          kv_cache_shape)
                    present = f'present_key_value_{idx}'
                    add_tensor(self.buffer[present], present)

                    if self.cross_attention:
                        cross_kv_cache_shape = (batch_size, 2,
                                                self.num_heads_kv, 0,
                                                self.head_size)
                        # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                        cross_kv_cache_buffer = torch.zeros((1, ),
                                                            dtype=torch.float32,
                                                            device=self.device)
                        add_tensor_with_shape(cross_kv_cache_buffer,
                                              f'cross_past_key_value_{idx}',
                                              cross_kv_cache_shape)
                        cross_present = f'cross_present_key_value_{idx}'
                        add_tensor(self.buffer[cross_present], cross_present)
                else:
                    key_value_cache = self.buffer[f'present_key_value_{idx}']
                    # when plugin is used, past_ket_value tensor does not need to be empty tensor
                    # because plugin does not care, and does not use this shape.
                    add_tensor(key_value_cache, f'past_key_value_{idx}')
                    add_tensor(key_value_cache, f'present_key_value_{idx}')

                    if self.cross_attention:
                        cross_cache_buffer = self.buffer[
                            f'cross_present_key_value_{idx}']
                        add_tensor(cross_cache_buffer,
                                   f'cross_past_key_value_{idx}')
                        add_tensor(cross_cache_buffer,
                                   f'cross_present_key_value_{idx}')

        if self.use_gpt_attention_plugin:
            # context request
            host_request_types = torch.zeros_like(context_lengths,
                                                  device='cpu').int()
            self.sequence_length_buffer = context_lengths.detach().clone()
            add_tensor_with_shape(self.sequence_length_buffer,
                                  'sequence_length', (batch_size, ))

            # field 0: past_key_value_length, field 1: is_context (deprecated). changed to [0], otherwise affects batch padded input mode
            add_tensor_with_shape(host_context_lengths,
                                  'host_past_key_value_lengths', (batch_size, ))
            add_tensor(host_request_types, 'host_request_types')
            for idx in range(self.first_layer, self.last_layer):
                add_tensor_with_shape(
                    self.host_max_attention_window_sizes[idx -
                                                         self.first_layer],
                    f'host_max_attention_window_size_{idx}', (1, ))
            if self.remove_input_padding:
                add_tensor(host_context_lengths, 'host_context_lengths')
        else:
            add_tensor(attention_mask, 'attention_mask')

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            add_tensor(self.all_reduce_workspace, 'all_reduce_workspace')

        if self.use_lora_plugin:
            for idx in range(self.num_layers):
                for lora_module in self.lora_target_modules:
                    layer_idx = idx + self.first_layer
                    lora_ranks = f'{lora_module}_lora_ranks_{layer_idx}'
                    add_tensor(self.buffer[lora_ranks], lora_ranks)
                    lora_weights = f'{lora_module}_lora_weights_pointers_{layer_idx}'
                    add_tensor(self.buffer[lora_weights], lora_weights)

        return tensors

    def _get_next_step_shape_buffer(
            self,
            batch_size: int,
            beam_width: int,
            max_context_length: int,
            step: int,
            context_lengths: torch.Tensor,
            host_context_lengths: torch.Tensor,
            position_ids: torch.Tensor,
            last_token_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            cache_indirection: torch.Tensor,
            kv_cache_block_pointers: List[torch.Tensor],
            host_kv_cache_block_pointers: List[torch.Tensor],
            hidden_states_input: torch.Tensor = None,
            prompt_embedding_table: torch.Tensor = None,
            tasks: torch.Tensor = None,
            prompt_vocab_size: torch.Tensor = None,
            encoder_output: torch.Tensor = None,
            encoder_input_lengths: torch.Tensor = None):
        tensors = {}  # Dict[str, RuntimeTensor]
        sym = lambda x, name: RuntimeTensor.from_torch(name, x)
        add_tensor = lambda x, name: tensors.update({name: sym(x, name)})
        add_tensor_with_shape = lambda x, name, shape: tensors.update(
            {name: RuntimeTensor.from_torch(name, x, override_shape=shape)})
        if self.use_gpt_attention_plugin:
            add_tensor(context_lengths, 'context_lengths')
        add_tensor(cache_indirection, 'cache_indirection')

        if self.mapping.has_pp():
            hidden_size = self.hidden_size * self.mapping.tp_size
            shape = (1, batch_size * beam_width,
                     hidden_size) if self.remove_input_padding else (
                         batch_size * beam_width, 1, hidden_size)
            hidden_states_input = hidden_states_input.resize_(*shape)

        if self.mapping.is_last_pp_rank():
            add_tensor(self.buffer['logits'], 'logits')

            if not self.gather_all_token_logits:
                add_tensor(last_token_ids, 'last_token_ids')
        else:
            add_tensor(hidden_states_input, 'hidden_states_output')

        if self.mapping.is_first_pp_rank():
            input_ids_shape = (1, batch_size *
                               beam_width) if self.remove_input_padding else (
                                   batch_size * beam_width, 1)
            add_tensor_with_shape(self.new_tokens, 'input_ids', input_ids_shape)
        else:
            add_tensor(hidden_states_input, 'hidden_states_input')

        if self.remove_input_padding:
            add_tensor(host_context_lengths, 'host_context_lengths')

        if self.has_position_embedding:
            add_tensor(position_ids, 'position_ids')

        if self.cross_attention:
            # hack: disable (or minimize) cross qkv computation at generation phase
            # TODO: enable [0,0,.] true zero tensor input; or use IfConditionalLayer
            encoder_output_shape = [1, 1, encoder_output.shape[-1]
                                    ]  # encoder_output.shape
            add_tensor_with_shape(encoder_output, 'encoder_output',
                                  encoder_output_shape)
            add_tensor(encoder_input_lengths, 'encoder_input_lengths')
            add_tensor(self.buffer['encoder_max_input_length'],
                       'encoder_max_input_length')

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer
                shape = kv_cache_block_pointers[idx].shape
                shape = [shape[0] * shape[1], *shape[2:]]
                add_tensor_with_shape(kv_cache_block_pointers[idx],
                                      f'kv_cache_block_pointers_{layer_idx}',
                                      shape)
                add_tensor_with_shape(
                    host_kv_cache_block_pointers[idx],
                    f'host_kv_cache_block_pointers_{layer_idx}', shape)

        if prompt_embedding_table is not None:
            add_tensor(prompt_embedding_table, 'prompt_embedding_table')

            if self.remove_input_padding:
                gen_tasks = tasks.unsqueeze(0)
            else:
                gen_tasks = tasks.unsqueeze(-1)
            add_tensor(gen_tasks, 'tasks')
            add_tensor(prompt_vocab_size, 'prompt_vocab_size')

        if not self.paged_kv_cache:
            for idx in range(self.first_layer, self.last_layer):
                if not self.use_gpt_attention_plugin:
                    next_shape = (batch_size * beam_width, 2, self.num_heads_kv,
                                  max_context_length + step, self.head_size)
                    if step % 2:
                        add_tensor_with_shape(
                            self.buffer[f'1_present_key_value_{idx}'],
                            f'past_key_value_{idx}', next_shape)
                        add_tensor(self.buffer[f'present_key_value_{idx}'],
                                   f'present_key_value_{idx}')
                    else:
                        add_tensor_with_shape(
                            self.buffer[f'present_key_value_{idx}'],
                            f'past_key_value_{idx}', next_shape)
                        add_tensor(self.buffer[f'1_present_key_value_{idx}'],
                                   f'present_key_value_{idx}')
                else:
                    key_value_cache = self.buffer[f'present_key_value_{idx}']
                    add_tensor(key_value_cache, f'past_key_value_{idx}')
                    add_tensor(key_value_cache, f'present_key_value_{idx}')

                    if self.cross_attention:
                        cross_cache_buffer = self.buffer[
                            f'cross_present_key_value_{idx}']
                        add_tensor(cross_cache_buffer,
                                   f'cross_past_key_value_{idx}')
                        add_tensor(cross_cache_buffer,
                                   f'cross_present_key_value_{idx}')

        if self.use_gpt_attention_plugin:
            # generation requests
            host_request_types = torch.ones_like(context_lengths,
                                                 device='cpu').int()
            # previous [past_kv_length, is_context] has been deprecated. only past_kv_length should be given here
            # Note we should use max_context_length here to align to max -- but isn't this done in attn plugin's max_element() already?
            host_past_key_value_lengths = torch.tensor(
                [max_context_length + step] * (batch_size * beam_width),
                dtype=torch.int32,
                device='cpu')
            add_tensor(host_past_key_value_lengths,
                       'host_past_key_value_lengths')
            add_tensor(host_request_types, 'host_request_types')
            # Sequence lengths are not used in the context phase actually.
            add_tensor_with_shape(self.sequence_length_buffer,
                                  'sequence_length',
                                  (batch_size * beam_width, ))

            for idx in range(self.first_layer, self.last_layer):
                add_tensor_with_shape(
                    self.host_max_attention_window_sizes[idx -
                                                         self.first_layer],
                    f'host_max_attention_window_size_{idx}', (1, ))
            if self.remove_input_padding:
                add_tensor(host_context_lengths, 'host_context_lengths')
        else:
            add_tensor(attention_mask, 'attention_mask')

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            add_tensor(self.all_reduce_workspace, 'all_reduce_workspace')

        if self.use_lora_plugin:
            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer
                for lora_module in self.lora_target_modules:
                    lora_ranks = f'{lora_module}_lora_ranks_{layer_idx}'
                    add_tensor(self.buffer[lora_ranks], lora_ranks)
                    lora_module = f'{lora_module}_lora_weights_pointers_{layer_idx}'
                    add_tensor(self.buffer[lora_module], lora_module)

        return tensors

    def _prepare_context_inputs(self, batch_size, context_lengths,
                                host_context_lengths, use_gpt_attention_plugin,
                                remove_input_padding, **kwargs):

        last_token_ids = context_lengths.detach().clone()
        if use_gpt_attention_plugin:
            max_context_length = kwargs.pop('max_context_length')
            if remove_input_padding:
                position_ids = torch.unsqueeze(
                    torch.concat([
                        torch.arange(0,
                                     host_context_lengths[i],
                                     dtype=torch.int32,
                                     device='cuda') for i in range(batch_size)
                    ]), 0)
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.tensor(range(max_context_length),
                                            dtype=torch.int32,
                                            device='cuda').reshape(
                                                [1,
                                                 -1]).expand([batch_size, -1])
            ret = {'last_token_ids': last_token_ids}
        else:
            input_ids = kwargs.pop('input_ids')
            pad_id = kwargs.pop('pad_id', None)
            attention_mask = _prepare_attention_mask(input_ids, pad_id)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.int()

            ret = {
                'attention_mask': attention_mask,
                'last_token_ids': last_token_ids
            }

        if self.has_position_embedding:
            ret['position_ids'] = position_ids

        return ret

    def _prepare_generation_inputs(self, batch_size, context_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):
        last_token_ids = torch.ones_like(context_lengths)
        if use_gpt_attention_plugin:
            step = kwargs.pop('step')
            position_ids = context_lengths + step
            if remove_input_padding:
                position_ids = torch.unsqueeze(position_ids, 0)
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.unsqueeze(position_ids, 1)

            ret = {'last_token_ids': last_token_ids}
        else:
            attention_mask = kwargs.pop('attention_mask')
            num_beams = kwargs.pop('num_beams')
            attention_mask = torch.cat((attention_mask,
                                        attention_mask.new_ones(
                                            (batch_size * num_beams, 1))),
                                       dim=-1).contiguous()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids.int()

            ret = {
                'last_token_ids': last_token_ids,
                'attention_mask': attention_mask,
            }

        if self.has_position_embedding:
            ret['position_ids'] = position_ids

        return ret

    def pp_communicate_new_tokens(self, should_stop, cache_indir,
                                  sequence_length):
        if self.mapping.is_last_pp_rank():
            for pg in self.mapping.pp_group:
                if pg == self.mapping.rank:
                    continue
                should_stop = should_stop.to(self.device)
                self.nccl_comm.send(should_stop, pg)
                self.nccl_comm.send(cache_indir, pg)
                self.nccl_comm.send(sequence_length, pg)
            self.nccl_comm.send(self.new_tokens, self.mapping.pp_group[0])
        else:
            should_stop = torch.zeros(1, dtype=torch.bool, device=self.device)
            self.nccl_comm.recv(should_stop, self.mapping.pp_group[-1])
            self.nccl_comm.recv(cache_indir, self.mapping.pp_group[-1])
            self.nccl_comm.recv(sequence_length, self.mapping.pp_group[-1])
            if self.mapping.is_first_pp_rank():
                self.nccl_comm.recv(self.new_tokens, self.mapping.pp_group[-1])
        return should_stop

    def pp_communicate_final_output_ids(self, final_output_ids, batch_size,
                                        beam_width):
        if self.mapping.is_last_pp_rank():
            self.nccl_comm.send(final_output_ids, self.mapping.pp_group[0])
        elif self.mapping.is_first_pp_rank():
            final_output_ids = torch.zeros(
                (batch_size, beam_width, self.max_seq_length),
                dtype=torch.int32,
                device=self.device)
            self.nccl_comm.recv(final_output_ids, self.mapping.pp_group[-1])
        return final_output_ids

    def finalize_decoder(self,
                         context_lengths,
                         batch_size,
                         beam_width,
                         scfg,
                         in_progress=False):
        final_output_ids = None
        if self.mapping.is_last_pp_rank():
            # output shape of self.gather_tree: [batch_size, beam_width, output_len]
            beam_hyps_args = [
                self.beam_hyps_output_ids_tgt,
                self.beam_hyps_sequence_lengths_tgt,
                self.beam_hyps_cum_log_probs, self.beam_hyps_normed_scores,
                self.beam_hyps_log_probs, self.beam_hyps_min_normed_scores,
                self.beam_hyps_num_beams, self.beam_hyps_is_done
            ]
            if scfg.use_beam_hyps and in_progress:
                # self.gather_tree modifies these args.
                # In streaming mode, this results in incorrect decoding in the following steps.
                beam_hyps_args = copy.deepcopy(beam_hyps_args)

            final_output_ids = self.gather_tree(
                self.sequence_length_buffer, self.output_ids, self.parent_ids,
                self.end_ids, context_lengths, self.cum_log_probs,
                *beam_hyps_args, self.finished, self.length_penalty, batch_size,
                beam_width, self.max_seq_length, scfg.use_beam_hyps)

        # Communicate ranks in Pipeline Parallelism
        if self.mapping.has_pp():
            final_output_ids = self.pp_communicate_final_output_ids(
                final_output_ids, batch_size, beam_width)

        return final_output_ids

    def handle_per_step(
            self, cache_indirections: list, step: int, batch_size: int,
            max_context_length: int, beam_width: int, input_ids: torch.Tensor,
            hidden_states: torch.Tensor, scfg: SamplingConfig,
            kv_cache_block_pointers: list, host_kv_cache_block_pointers: list,
            prompt_embedding_table: torch.Tensor, tasks: torch.Tensor,
            context_lengths: torch.Tensor, host_context_lengths,
            attention_mask: torch.Tensor, prompt_vocab_size: torch.Tensor,
            ite: int, sequence_limit_lengths: torch.Tensor,
            sequence_lengths: torch.Tensor,
            next_step_tensors: Dict[str, RuntimeTensor], stop_words_list,
            bad_words_list, no_repeat_ngram_size, encoder_output: torch.Tensor,
            encoder_input_lengths: torch.Tensor,
            stopping_criteria: StoppingCriteria,
            logits_processor: LogitsProcessor):
        if step % 2:
            context = self.runtime.context_0
            this_src_cache_indirection = cache_indirections[1]
            this_tgt_cache_indirection = cache_indirections[0]
            next_src_cache_indirection = cache_indirections[0]
        else:
            context = self.runtime.context_1
            this_src_cache_indirection = cache_indirections[0]
            this_tgt_cache_indirection = cache_indirections[1]
            next_src_cache_indirection = cache_indirections[1]

        if step == 0:
            model_inputs = self._prepare_context_inputs(
                batch_size=batch_size,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                remove_input_padding=self.remove_input_padding,
                max_context_length=max_context_length,
                input_ids=input_ids,
                pad_id=scfg.pad_id,
                eos_id=scfg.end_id)

            position_ids = model_inputs.get('position_ids', None)
            last_token_ids = model_inputs.get('last_token_ids')
            attention_mask = model_inputs.get('attention_mask', None)

            if self.paged_kv_cache:
                host_kv_cache_block_pointers = self.kv_cache_manager.get_pointer_arrays(
                    1)
                kv_cache_block_pointers = [
                    x.to('cuda') for x in host_kv_cache_block_pointers
                ]

            ctx_tensors = self._get_context_shape_buffer(
                input_ids, context_lengths, host_context_lengths, position_ids,
                last_token_ids, attention_mask, this_src_cache_indirection,
                kv_cache_block_pointers, host_kv_cache_block_pointers,
                hidden_states, prompt_embedding_table, tasks, prompt_vocab_size,
                encoder_output, encoder_input_lengths)
            context = self.runtime.ctx_context
            self.runtime._set_tensors(context, ctx_tensors)
            if self.debug_mode:
                self.debug_buffer = {
                    name: tensor.to_torch()
                    for name, tensor in ctx_tensors.items()
                }
            if self.cuda_graph_mode:
                # context mode, clean cuda graph instances
                self.runtime.cuda_graph_instances = [None for _ in range(2)]

        # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
        stream = torch.cuda.current_stream().cuda_stream
        instance_idx = step % 2
        if self.cuda_graph_mode and self.runtime.cuda_graph_instances[
                instance_idx] is not None:
            # launch cuda graph
            CUASSERT(
                cudart.cudaGraphLaunch(
                    self.runtime.cuda_graph_instances[instance_idx], stream))
            ok = True
        else:
            ok = self.runtime._run(context, stream)

        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        if self.debug_mode:
            torch.cuda.synchronize()

        context_logits = None
        if self.mapping.is_last_pp_rank():
            if step == 0 and self.gather_all_token_logits:
                context_logits = self.buffer['logits'].detach().clone()
                if self.remove_input_padding:
                    # reshape self.buffer['logits'] from [bs, max_context_length, vocab]
                    # to [1, bs * max_context_length, vocab]
                    # Note that the data are put in the buffer without padding although
                    # the allocated buffer has padding.
                    self.buffer['logits'] = self.buffer['logits'].reshape(
                        [1, -1, self.buffer['logits'].shape[-1]])
                    self.buffer['logits'] = torch.index_select(
                        self.buffer['logits'], 1,
                        last_token_ids - 1).view(batch_size,
                                                 self.vocab_size_padded)
                else:
                    last_token_ids = last_token_ids.reshape(batch_size, 1, 1)
                    last_token_ids = last_token_ids.expand(
                        batch_size, 1, self.vocab_size_padded) - 1
                    self.buffer['logits'] = torch.gather(
                        self.buffer['logits'],
                        dim=1,
                        index=last_token_ids.to(dtype=torch.int64)).view(
                            batch_size, self.vocab_size_padded)

        if step == 0 and beam_width > 1:
            # these tiled tensors are returned by handle_per_step(), so they can relay to the next generation calls
            if not self.use_gpt_attention_plugin:
                attention_mask = _tile_beam_width(attention_mask, beam_width)
            context_lengths = _tile_beam_width(context_lengths, beam_width)
            host_context_lengths = _tile_beam_width(host_context_lengths,
                                                    beam_width)
            if encoder_input_lengths is not None:
                encoder_input_lengths = _tile_beam_width(
                    encoder_input_lengths, beam_width)

            if tasks is not None:
                tasks = _tile_beam_width(tasks, beam_width)

            # Move tiling before logit computing of context
            if not self.paged_kv_cache:
                for key in self.buffer.keys():
                    # Note: this tiles both self attn cache and cross attn cache!
                    # both names contain "present_key_value"
                    if "present_key_value" in key:
                        self.buffer[key] = _tile_beam_width(
                            self.buffer[key], beam_width)
            if self.mapping.is_last_pp_rank():
                self.buffer['logits'] = _tile_beam_width(
                    self.buffer['logits'], beam_width)

        # Initialize sequence_lengths (no paddings) for the generation phase.
        if step == 0:
            self.sequence_length_buffer = context_lengths.detach().clone()

        if not step == self.max_new_tokens - 1:
            # Set shape and address for the next step
            model_inputs = self._prepare_generation_inputs(
                batch_size=batch_size,
                context_lengths=context_lengths,
                use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                remove_input_padding=self.remove_input_padding,
                step=step,
                num_beams=beam_width,
                attention_mask=attention_mask,
            )

            position_ids = model_inputs.get('position_ids', None)
            last_token_ids = model_inputs.get('last_token_ids')
            attention_mask = model_inputs.get('attention_mask', None)

            # Prepare for the next step, and always allocate 1 token slot.
            if self.paged_kv_cache:
                # Iterate to the next step in KV cache manager.
                # Increase number of tokens for all unfinished sequences.
                # And allocate new blocks if needed.
                # We set this to False for all sequences, since we use only length criterion to stop now
                self.kv_cache_manager.step([False] * batch_size)
                host_kv_cache_block_pointers = self.kv_cache_manager.get_pointer_arrays(
                    beam_width)
                kv_cache_block_pointers = [
                    x.to('cuda') for x in host_kv_cache_block_pointers
                ]

            next_context = self.runtime.context_1 if step % 2 else self.runtime.context_0
            next_step_tensors = self._get_next_step_shape_buffer(
                batch_size, beam_width, max_context_length, step,
                context_lengths, host_context_lengths, position_ids,
                last_token_ids, attention_mask, next_src_cache_indirection,
                kv_cache_block_pointers, host_kv_cache_block_pointers,
                hidden_states, prompt_embedding_table, tasks, prompt_vocab_size,
                encoder_output, encoder_input_lengths)
            # there are some tensors created inside the _get_next_step_shape_buffer, not owned by any object
            # needs to pro-long the life time of the tensors inside the next_step_tensors array
            # otherwise, it maybe released before the next step actually enqueued
            # one way to prolong it is to return the list, and destroy it in next step by assigning new values
            self.runtime._set_tensors(next_context, next_step_tensors)

            if self.debug_mode:
                self.debug_buffer = {
                    name: tensor.to_torch()
                    for name, tensor in next_step_tensors.items()
                }
            if self.cuda_graph_mode:
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(
                        stream, cudart.cudaStreamCaptureMode.
                        cudaStreamCaptureModeGlobal))
                next_context.execute_async_v3(stream)
                next_graph = CUASSERT(cudart.cudaStreamEndCapture(stream))[0]

                instance_idx = (step + 1) % 2

                if self.runtime.cuda_graph_instances[instance_idx] is not None:
                    self.runtime.cuda_graph_instances[
                        instance_idx] = _update_cuda_graph_instance(
                            self.runtime.cuda_graph_instances[instance_idx],
                            next_graph)
                else:
                    self.runtime.cuda_graph_instances[instance_idx] = CUASSERT(
                        cudart.cudaGraphInstantiate(next_graph, 0))[0]

                # Pre-upload cuda graph to stream
                CUASSERT(
                    cudart.cudaGraphUpload(
                        self.runtime.cuda_graph_instances[instance_idx],
                        stream))

        should_stop = None
        logits = None
        if self.mapping.is_last_pp_rank():
            logits = self.buffer['logits']
            if self.debug_mode:
                for k in self.debug_buffer:
                    # if needed, apply filter based on output name
                    tensors_to_save = self.debug_tensors
                    if self.debug_tensors_to_save is not None:
                        tensors_to_save = self.debug_tensors_to_save
                    if all([kk not in k for kk in tensors_to_save]):
                        continue
                    t = self.debug_buffer[k]
                    t = t.view(-1, t.shape[-1])  # consolidate all but last dim
                    # convert tensor name to valid file name
                    fname = "".join(c for c in k if (c.isalnum() or c in "._-"))
                    np.savetxt(f"{fname}-step{step}.txt", t.cpu().detach())
            if logits is not None:
                if logits_processor is not None:
                    final_output_ids = self.finalize_decoder(context_lengths,
                                                             batch_size,
                                                             beam_width,
                                                             scfg,
                                                             in_progress=True)
                    # keep the shape as same as huggingface stopping_criteria
                    final_output_ids_ = final_output_ids.reshape(
                        -1, final_output_ids.size(-1))
                    logits = logits_processor(step, final_output_ids_, logits)
                    self.buffer['logits'] = logits
                # [batch_size x beam_width, vocab_size_padded] -> [batch_size, beam_width, vocab_size_padded]
                next_token_logits = logits.reshape(
                    (batch_size, beam_width, -1)).to(self.decoder_logits_dtype)
                decode_step = step + max_context_length

                should_stop = self.dynamic_decoder.forward(
                    next_token_logits, decode_step, max_context_length,
                    self.max_attention_window_size, ite, batch_size,
                    self.end_ids, self.embedding_bias_opt, context_lengths,
                    sequence_limit_lengths, stop_words_list, bad_words_list,
                    no_repeat_ngram_size, this_src_cache_indirection,
                    self.output_ids, self.new_tokens, self.finished,
                    self.finished, self.sequence_length_buffer,
                    self.cum_log_probs, self.log_probs, self.parent_ids,
                    this_tgt_cache_indirection, self.beam_hyps_output_ids_tgt,
                    self.beam_hyps_sequence_lengths_tgt,
                    self.beam_hyps_cum_log_probs, self.beam_hyps_normed_scores,
                    self.beam_hyps_log_probs, self.beam_hyps_min_normed_scores,
                    self.beam_hyps_num_beams, self.beam_hyps_is_done,
                    scfg.use_beam_hyps)
                if stopping_criteria is not None and not should_stop.item():
                    final_output_ids = self.finalize_decoder(context_lengths,
                                                             batch_size,
                                                             beam_width,
                                                             scfg,
                                                             in_progress=True)
                    # keep the shape as same as huggingface stopping_criteria
                    final_output_ids_ = final_output_ids.reshape(
                        -1, final_output_ids.size(-1))
                    should_stop[0] = stopping_criteria(step, final_output_ids_,
                                                       logits)

        if self.mapping.has_pp():
            should_stop = self.pp_communicate_new_tokens(
                should_stop, this_tgt_cache_indirection,
                self.sequence_length_buffer)

        if self.paged_kv_cache:
            if (step >= self.max_new_tokens - 1) or (should_stop is not None
                                                     and should_stop.item()):
                # Free all blocks in all sequences.
                # With in-flight batching and while loop we'll free some sequences, when they are done
                self.kv_cache_manager.step([True] * batch_size)

        return should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, context_logits, encoder_input_lengths

    def decode_regular(self,
                       batch_size: int,
                       scfg: SamplingConfig,
                       sequence_lengths: torch.Tensor,
                       context_lengths: torch.Tensor,
                       host_context_lengths,
                       max_context_length: int,
                       beam_width: int,
                       cache_indirections: list,
                       input_ids: torch.Tensor,
                       hidden_states: torch.Tensor,
                       prompt_embedding_table: torch.Tensor,
                       tasks: torch.Tensor,
                       prompt_vocab_size: torch.Tensor,
                       ite: int,
                       sequence_limit_lengths: torch.Tensor,
                       stop_words_list,
                       bad_words_list,
                       no_repeat_ngram_size,
                       output_sequence_lengths: bool = False,
                       return_dict: bool = False,
                       encoder_output: torch.Tensor = None,
                       encoder_input_lengths: torch.Tensor = None,
                       stopping_criteria: StoppingCriteria = None,
                       logits_processor: LogitsProcessor = None,
                       **kwargs):
        kv_cache_block_pointers = []
        host_kv_cache_block_pointers = []
        attention_mask = None
        context_logits = None
        generation_logits = []

        def get_outputs_dict(output_ids):
            outputs = {}
            outputs['output_ids'] = output_ids
            if output_sequence_lengths:
                outputs[
                    'sequence_lengths'] = self.sequence_length_buffer.reshape(
                        [batch_size, beam_width])
            if self.gather_all_token_logits:
                outputs['context_logits'] = context_logits
                outputs['generation_logits'] = generation_logits
            return outputs

        benchmark_profiler = kwargs.get('benchmark_profiler', None)
        generation_phase_step_count = 0

        def profile_fn(benchmark_profiler_obj, step_count):
            if benchmark_profiler_obj is not None:
                benchmark_profiler_obj.record_cuda_event('last_token')
                benchmark_profiler_obj.record_elapsed_time(
                    'first_token', 'last_token', 'generation_time')
                benchmark_profiler_obj.add_aux_info('generation_step_count',
                                                    step_count)

        next_step_tensors = None
        for step in range(0, self.max_new_tokens):
            should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, logits, encoder_input_lengths = self.handle_per_step(
                cache_indirections, step, batch_size, max_context_length,
                beam_width, input_ids, hidden_states, scfg,
                kv_cache_block_pointers, host_kv_cache_block_pointers,
                prompt_embedding_table, tasks, context_lengths,
                host_context_lengths, attention_mask, prompt_vocab_size, ite,
                sequence_limit_lengths, sequence_lengths, next_step_tensors,
                stop_words_list, bad_words_list, no_repeat_ngram_size,
                encoder_output, encoder_input_lengths, stopping_criteria,
                logits_processor)
            if step == 0:
                if benchmark_profiler is not None:
                    benchmark_profiler.record_cuda_event('first_token')
            else:
                generation_phase_step_count = generation_phase_step_count + 1
            if self.gather_all_token_logits:
                if self.mapping.is_last_pp_rank():
                    if step == 0:
                        context_logits = logits
                    else:
                        l = next_step_tensors['logits'].to_torch()
                        generation_logits.append(l.clone().detach())

            if should_stop is not None and should_stop.item():
                profile_fn(benchmark_profiler, generation_phase_step_count)
                final_output_ids = self.finalize_decoder(
                    context_lengths, batch_size, beam_width, scfg)

                if self.mapping.is_first_pp_rank():
                    if return_dict:
                        return get_outputs_dict(final_output_ids)
                    else:
                        return final_output_ids
                elif self.mapping.is_last_pp_rank(
                ) and self.gather_all_token_logits:
                    outputs = {}
                    outputs['context_logits'] = context_logits
                    outputs['generation_logits'] = generation_logits
                    return outputs
                else:
                    return None

        profile_fn(benchmark_profiler, generation_phase_step_count)

        final_output_ids = self.finalize_decoder(context_lengths, batch_size,
                                                 beam_width, scfg)

        if self.mapping.is_first_pp_rank():
            if return_dict:
                return get_outputs_dict(final_output_ids)
            else:
                return final_output_ids
        elif self.mapping.is_last_pp_rank() and self.gather_all_token_logits:
            outputs = {}
            outputs['context_logits'] = context_logits
            outputs['generation_logits'] = generation_logits
            return outputs
        else:
            return None

    def decode_stream(self,
                      batch_size: int,
                      scfg: SamplingConfig,
                      sequence_lengths: torch.Tensor,
                      context_lengths: torch.Tensor,
                      host_context_lengths,
                      max_context_length: int,
                      beam_width: int,
                      cache_indirections: list,
                      input_ids: torch.Tensor,
                      hidden_states: torch.Tensor,
                      prompt_embedding_table: torch.Tensor,
                      tasks: torch.Tensor,
                      prompt_vocab_size: torch.Tensor,
                      ite: int,
                      sequence_limit_lengths: torch.Tensor,
                      stop_words_list,
                      bad_words_list,
                      no_repeat_ngram_size,
                      output_sequence_lengths: bool = False,
                      return_dict: bool = False,
                      encoder_output: torch.Tensor = None,
                      encoder_input_lengths: torch.Tensor = None,
                      stopping_criteria: StoppingCriteria = None,
                      logits_processor: LogitsProcessor = None,
                      **kwargs):
        kv_cache_block_pointers = []
        host_kv_cache_block_pointers = []
        attention_mask = None
        context_logits = None

        def get_outputs_dict(output_ids):
            outputs = {}
            outputs['output_ids'] = output_ids
            if output_sequence_lengths:
                outputs[
                    'sequence_lengths'] = self.sequence_length_buffer.reshape(
                        [batch_size, beam_width])
            if self.gather_all_token_logits:
                outputs['context_logits'] = context_logits
            return outputs

        next_step_tensors = None
        for step in range(0, self.max_new_tokens):
            should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, logits, encoder_input_lengths = self.handle_per_step(
                cache_indirections, step, batch_size, max_context_length,
                beam_width, input_ids, hidden_states, scfg,
                kv_cache_block_pointers, host_kv_cache_block_pointers,
                prompt_embedding_table, tasks, context_lengths,
                host_context_lengths, attention_mask, prompt_vocab_size, ite,
                sequence_limit_lengths, sequence_lengths, next_step_tensors,
                stop_words_list, bad_words_list, no_repeat_ngram_size,
                encoder_output, encoder_input_lengths, stopping_criteria,
                logits_processor)
            if step == 0:
                context_logits = logits
            if should_stop is not None:

                final_output_ids = self.finalize_decoder(context_lengths,
                                                         batch_size,
                                                         beam_width,
                                                         scfg,
                                                         in_progress=True)

                if self.mapping.is_first_pp_rank():
                    if return_dict:
                        yield get_outputs_dict(final_output_ids)
                    else:
                        yield final_output_ids
                else:
                    yield None

                if should_stop.item():
                    return

        final_output_ids = self.finalize_decoder(context_lengths, batch_size,
                                                 beam_width, scfg)
        if self.mapping.is_first_pp_rank():
            if return_dict:
                yield get_outputs_dict(final_output_ids)
            else:
                yield final_output_ids
        else:
            yield None

    def decode_batch(self,
                     input_ids: Sequence[torch.Tensor],
                     sampling_config: SamplingConfig,
                     streaming: bool = False,
                     **kwargs):
        input_ids, context_lengths = _prepare_input_ids(input_ids)
        return self.decode(input_ids,
                           context_lengths,
                           sampling_config,
                           streaming=streaming,
                           **kwargs)

    # As dynamic_decoder uses torch's current stream, we must ensure it runs on the same stream that
    # dynamic_decoder was set up with
    @cuda_stream_guard
    def decode(self,
               input_ids: torch.Tensor,
               context_lengths: torch.Tensor,
               sampling_config: SamplingConfig,
               prompt_embedding_table: torch.Tensor = None,
               tasks: torch.Tensor = None,
               prompt_vocab_size: torch.Tensor = None,
               stop_words_list=None,
               bad_words_list=None,
               no_repeat_ngram_size=None,
               streaming: bool = False,
               output_sequence_lengths: bool = False,
               return_dict: bool = False,
               encoder_output: torch.Tensor = None,
               encoder_input_lengths: torch.Tensor = None,
               stopping_criteria: StoppingCriteria = None,
               logits_processor: LogitsProcessor = None,
               **kwargs):
        scfg = sampling_config
        batch_size = context_lengths.size(0)
        beam_width = scfg.num_beams
        max_context_length = torch.max(context_lengths).item()
        host_context_lengths = context_lengths.cpu()
        assert batch_size == self.batch_size, \
            "Given batch size is different from the one used in setup()," \
            "rerun the setup function with the new batch size to avoid buffer overflow."
        assert max_context_length == self.max_context_length, \
            "Given input length is large then the one used in setup()," \
            "rerun the setup function with the new max_context_length to avoid buffer overflow."
        assert beam_width == self.beam_width, \
            "Given beam width is different from the one used in setup()," \
            "rerun the setup function with the new beam width to avoid buffer overflow."
        ite = 0  # index of local batches, will always be 0 if pp_size = 1

        self.__setup_decoder(input_ids, scfg, host_context_lengths)
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        sequence_limit_lengths = torch.full((batch_size, 1),
                                            self.max_seq_length,
                                            dtype=torch.int32,
                                            device=self.device)

        # Sequence_lengths for the dynamic decoder still has the input paddings.
        sequence_lengths = torch.full((batch_size * beam_width, 1),
                                      max_context_length,
                                      dtype=torch.int32,
                                      device=self.device)

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                self.max_attention_window_size,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device),
            torch.full((
                batch_size,
                beam_width,
                self.max_attention_window_size,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device)
        ]  # ping-pong buffers

        hidden_states = None
        if self.mapping.has_pp():
            max_num_tokens = max(batch_size * beam_width,
                                 batch_size * self.max_seq_length)
            hidden_size = self.hidden_size * self.mapping.tp_size
            hidden_states = torch.zeros((1, max_num_tokens, hidden_size))

        # Init KV cache block manager
        if self.paged_kv_cache:
            max_blocks_per_seq = math.ceil(self.max_attention_window_size /
                                           self.tokens_per_block)
            blocks = batch_size * beam_width * max_blocks_per_seq
            memory_pools = [
                self.buffer[f'present_key_value_{i}']
                for i in range(self.first_layer, self.last_layer)
            ]
            self.kv_cache_manager = KVCacheManager(
                memory_pools, blocks, self.tokens_per_block, max_blocks_per_seq,
                self.max_attention_window_size, beam_width)

            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                self.kv_cache_manager.add_sequence(generation_sequence,
                                                   max_context_length)

        # start context phase
        if streaming:
            return self.decode_stream(
                batch_size, scfg, sequence_lengths, context_lengths,
                host_context_lengths, max_context_length, beam_width,
                cache_indirections, input_ids, hidden_states,
                prompt_embedding_table, tasks, prompt_vocab_size, ite,
                sequence_limit_lengths, stop_words_list, bad_words_list,
                no_repeat_ngram_size, output_sequence_lengths, return_dict,
                encoder_output, encoder_input_lengths, stopping_criteria,
                logits_processor, **kwargs)
        else:
            return self.decode_regular(
                batch_size, scfg, sequence_lengths, context_lengths,
                host_context_lengths, max_context_length, beam_width,
                cache_indirections, input_ids, hidden_states,
                prompt_embedding_table, tasks, prompt_vocab_size, ite,
                sequence_limit_lengths, stop_words_list, bad_words_list,
                no_repeat_ngram_size, output_sequence_lengths, return_dict,
                encoder_output, encoder_input_lengths, stopping_criteria,
                logits_processor, **kwargs)


class ChatGLMGenerationSession(GenerationSession):

    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer,
        mapping: Mapping,
        debug_mode=False,
        debug_tensors_to_save=None,
        cuda_graph_mode=False,
        stream: torch.cuda.Stream = None,
    ):

        super().__init__(
            model_config,
            engine_buffer,
            mapping,
            debug_mode,
            debug_tensors_to_save,
            cuda_graph_mode,
            stream,
        )

        self.mask_index_tensor = None

    def _prepare_context_inputs(self, batch_size, context_lengths,
                                use_gpt_attention_plugin, remove_input_padding,
                                **kwargs):

        max_context_length = kwargs.pop('max_context_length')
        last_token_ids = context_lengths.detach().clone()

        if remove_input_padding:
            input_lengths_acc = torch.cumsum(torch.cat(
                [torch.IntTensor([0]).cuda(), context_lengths], dim=0),
                                             dim=0)
            position_ids = torch.zeros([1, 2, input_lengths_acc[-1]],
                                       dtype=torch.int32)
            for i in range(batch_size):
                position_ids[0, 0, input_lengths_acc[i]:input_lengths_acc[
                    i + 1]] = torch.arange(0,
                                           context_lengths[i],
                                           dtype=torch.int32)
                position_ids[0, 0, input_lengths_acc[i + 1] -
                             1] = context_lengths[i] - 2
                position_ids[0, 1, input_lengths_acc[i + 1] - 1] = 1
            position_ids = position_ids.int().cuda()
            last_token_ids = torch.cumsum(last_token_ids, dim=0).int().cuda()
        else:
            position_ids = torch.zeros([batch_size, 2, max_context_length],
                                       dtype=torch.int32)
            position_ids[:, 0, :] = torch.arange(max_context_length)

            if kwargs["pad_id"] == 50256:  # specialization for GLM-10B
                self.mask_index_tensor = torch.zeros([batch_size],
                                                     dtype=torch.int32)
                for i in range(batch_size):
                    length = context_lengths[i]
                    mask_index = torch.where(
                        kwargs["input_ids"][i] == 50260)[0].int()
                    gmask_index = torch.where(
                        kwargs["input_ids"][i] == 50263)[0].int()
                    smask_index = torch.where(
                        kwargs["input_ids"][i] == 50264)[0].int()
                    tail_index = torch.Tensor([max_context_length]).int().cuda()
                    mask_index = torch.cat([
                        mask_index.int(), gmask_index, smask_index, tail_index
                    ],
                                           dim=0).min()
                    position_ids[i, 0, max_context_length - 1] = int(mask_index)
                    position_ids[i, 1, max_context_length - 1] = 1
                    self.mask_index_tensor[i] = int(mask_index)
            else:
                for i in range(batch_size):
                    length = context_lengths[i]
                    position_ids[i, 0, length - 1] = length - 2
                    position_ids[i, 1, length - 1] = 1

            position_ids = position_ids.cuda()

        inputs = {
            'position_ids': position_ids,
            'last_token_ids': last_token_ids
        }
        if not use_gpt_attention_plugin:
            attention_mask = torch.zeros((batch_size, 1))
            inputs['attention_mask'] = attention_mask
        return inputs

    def _prepare_generation_inputs(self, batch_size, context_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):

        step = kwargs.pop('step')
        num_beams = kwargs.pop('num_beams')
        last_token_ids = torch.ones_like(context_lengths)

        if remove_input_padding:
            position_ids = torch.zeros([1, 2, batch_size], dtype=torch.int32)
            for i in range(batch_size):
                position_ids[0, 0, i] = context_lengths[i * num_beams] - 2
                position_ids[0, 1, i] = step + 2
            position_ids = _tile_beam_width(position_ids, num_beams)
            position_ids = position_ids.int().cuda()
            last_token_ids = torch.cumsum(last_token_ids, dim=0).int().cuda()
        else:
            data = []
            if self.mask_index_tensor is not None:  # specialization for GLM-10B
                for i in range(batch_size):
                    data.append([[self.mask_index_tensor[i]], [step + 2]])
            else:
                for i in range(batch_size):
                    data.append([[context_lengths[i * num_beams] - 2],
                                 [step + 2]])
            position_ids = torch.tensor(data, dtype=torch.int32, device='cuda')
            position_ids = _tile_beam_width(position_ids, num_beams)

        inputs = {
            'position_ids': position_ids,
            'last_token_ids': last_token_ids
        }
        if not use_gpt_attention_plugin:
            attention_mask = torch.zeros((batch_size, 1))
            inputs['attention_mask'] = attention_mask
        return inputs


class QWenForCausalLMGenerationSession(GenerationSession):

    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer,
        mapping: Mapping,
        debug_mode=False,
        debug_tensors_to_save=None,
        cuda_graph_mode=False,
        stream: torch.cuda.Stream = None,
        global_max_input_length: int = 2048,
        global_max_output_length: int = 4096,
    ):
        super().__init__(model_config,
                         engine_buffer,
                         mapping,
                         debug_mode,
                         debug_tensors_to_save=debug_tensors_to_save,
                         cuda_graph_mode=cuda_graph_mode,
                         stream=stream)
        self.global_max_input_length = global_max_input_length
        self.global_max_output_length = global_max_output_length

    def generate(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        sampling_config: SamplingConfig,
        max_new_tokens: int,
        runtime_rank: int = 0,
    ):
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(max_new_tokens,
                             self.global_max_output_length - max_input_length)
        # setup batch_size, max_input_length, max_output_len
        self.setup(batch_size=input_lengths.size(0),
                   max_context_length=max_input_length,
                   max_new_tokens=max_new_tokens)
        output_ids = self.decode(input_ids, input_lengths, sampling_config)
        with torch.no_grad():
            torch.cuda.synchronize()
            if runtime_rank == 0:
                outputs = output_ids[:, 0, :]
                return outputs
