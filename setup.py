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
import os
import platform
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.dist import Distribution


def parse_requirements(filename: os.PathLike):
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != '-', line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith("#") or line.startswith("-r"):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            else:
                deps.append(line)
    return deps, extra_URLs


def get_version():
    version_file = Path(
        __file__).resolve().parent / "tensorrt_llm" / "version.py"
    version = None
    with open(version_file) as f:
        for line in f:
            if not line.startswith("__version__"):
                continue
            version = line.split('"')[1]

    if version is None:
        raise RuntimeError(f"Could not set version from {version_file}")

    return version


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


on_windows = platform.system() == "Windows"
required_deps, extra_URLs = parse_requirements(
    Path("requirements-windows.txt" if on_windows else "requirements.txt"))
devel_deps, _ = parse_requirements(
    Path("requirements-dev-windows.txt"
         if on_windows else "requirements-dev.txt"))

setup(
    name='tensorrt_llm',
    version=get_version(),
    description='TensorRT-LLM: A TensorRT Toolbox for Large Language Models',
    install_requires=required_deps,
    dependency_links=extra_URLs,
    zip_safe=True,
    license="Apache License 2.0",
    packages=find_packages(),
    # TODO Add windows support for python bindings.
    package_data={
        'tensorrt_llm': ([
            'libs/th_common.dll', 'libs/nvinfer_plugin_tensorrt_llm.dll',
            'bindings.*.pyd'
        ] if platform.system() == "Windows" else [
            'libs/libth_common.so',
            'libs/libhackNCCL.so',
            'libs/libnvinfer_plugin_tensorrt_llm.so',
            'bindings.*.so',
        ]) + ['bindings/*.pyi', 'tools/plugin_gen/templates/*'],
    },
    entry_points={
        'console_scripts': ['trtllm-build=tensorrt_llm.commands.build:main'],
    },
    python_requires=">=3.7, <4",
    distclass=BinaryDistribution,
    extras_require={"devel": devel_deps},
)
