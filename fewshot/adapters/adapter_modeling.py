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

import torch.nn as nn
from .utils import Activations

class Adapter(nn.Module):
    """Conventional adapter latyer."""
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.activation = Activations(config.nonlinearity.lower())
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)

    def forward(self, x):
        output = self.down_sampler(x)
        output = self.activation(output)
        return self.up_sampler(output)