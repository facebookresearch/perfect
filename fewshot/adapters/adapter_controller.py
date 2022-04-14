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

"""Implements adapter controller, a module that apply adapter layers."""
import torch.nn as nn 
from .adapter_modeling import Adapter

class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logits of
    putting adapter layers within  the transformer's layers.
    config: adapter configuraiton.
    input_dim: input dimension of the hidden representation feed into adapters."""
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.adapter = self.construct_adapters()
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(input_dim)

    def construct_adapters(self):
        """Construct the Adapter layers."""
        return Adapter(self.config, input_dim=self.input_dim)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z) 
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs 
