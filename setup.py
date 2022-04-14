# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# This file is part of PERFECT.
# See https://github.com/facebookresearch/perfect for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install Making Language models better few-shot learners."""
import setuptools


def setup_package():
    setuptools.setup(
        name='fewshot',
        version='0.0.1',
        description='PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models',
        author='Rabeeh Karimi Mahabadi',
        license='MIT License',
        packages=setuptools.find_packages(
            exclude=['docs', 'tests', 'scripts', 'examples']),
        install_requires=[
        ],
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
        ],
        keywords='nlp machinelearning',
    )


if __name__ == '__main__':
    setup_package()
