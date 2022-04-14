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