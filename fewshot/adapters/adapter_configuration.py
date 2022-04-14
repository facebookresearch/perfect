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

"""Implements the adapter's configuration."""
from dataclasses import dataclass 

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    # This is for the layernorms applied after feedforward/self-attention layers.
    add_layer_norm_before_adapter: bool = False  
    add_layer_norm_after_adapter: bool = False 
    nonlinearity: str = "gelu_new"
    reduction_factor: int = 16
    # By default, we add adapters after attention, set False if otherwise.
    add_adapter_after_attention = True
    add_adapter_after_feedforward = True
    # Trains the adapters if this is set to true.
    adapter_tune = False   
