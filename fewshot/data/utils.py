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

"""Defines the utilities to process the datasets."""
import string 
import torch

from dataclasses import dataclass
from typing import Optional
from transformers.file_utils import ModelOutput


@dataclass
class Text(ModelOutput):
    shortenable: Optional[torch.BoolTensor] = False  
    text: str = None


def get_verbalization_ids(word, tokenizer):
    """Tokenize a verbalization word and return the tokens."""
    return tokenizer.encode(word, add_special_tokens=False)


def remove_final_punctuation(word):
    return word.rstrip(string.punctuation)
    
def lowercase(word):
    return word.lower()