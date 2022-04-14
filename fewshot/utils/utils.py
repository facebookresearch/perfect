"""Implements utility functions."""
import os
import json 
import numpy as np 

import torch
import torch.nn as nn

from fewshot.adapters import AdapterConfig
from fewshot.adapters import AdapterController

def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
def save_json(results, filepath):
    """Saves results in a json format to a file."""
    with open(filepath, 'w') as f:
        json.dump(results, f)

def load_json(filepath):
    """Loads a jason file into a dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)

def write_lines(path, lines):
    """Write lines into a file."""
    with open(path, 'w') as f:
      for line in lines:
          f.write(line+"\n")

def compute_accuracy_from_losses(losses, targets):
    """Computes the accuracy from the given loss.
    losses: Is a list of list of size number of targets.
    targets: shows the ground truth labels.
    Predicted labels are counted as the label associated
    with the minimum loss.""" 
    predictions = np.argmin(losses, axis=0)
    accuracy = np.mean(predictions==targets)
    return accuracy, predictions

def get_adapter_config(args):
    """Creates the Adapter configurations from the given
    inputs arguments."""
    adapter_config = AdapterConfig()
    for arg in vars(args):
        if hasattr(adapter_config, arg):
           setattr(adapter_config, arg, getattr(args, arg))
    return adapter_config

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def set_layernorms_trainable_params(model, tune_layernorms):
    for n, sub_module in model.named_modules():
        if isinstance(sub_module, nn.LayerNorm):
            for n, p in sub_module.named_parameters():
                p.requires_grad = tune_layernorms
 
def set_trainable_params_for_adapters(model, tune_layernorms):
    """Freezes all the parameters of the model, 
    except for the adapters, and layernorms if specified."""
    freeze_model(model)
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, AdapterController) or name =="extra_embeddings":
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    set_layernorms_trainable_params(model, tune_layernorms)

def set_trainable_params_for_bitfit(model, bitfit_tune_lm_head):
    freeze_model(model)
    for n, p in model.named_parameters():
        if ".bias" in n:
            p.requires_grad = True
    if bitfit_tune_lm_head:
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = True

    # Set label embeddings to true.
    for name, sub_module in model.named_modules():
        if name =="extra_embeddings":
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True

def set_trainable_params_for_prompt_tuning(model):
    freeze_model(model)
    # Set prompt and label embeddings to true.
    for name, sub_module in model.named_modules():
        if name =="extra_embeddings" or name=="prompt_embedding":
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True

def set_config_args(config, args):
    """Sets the pruning arguments in the config."""
    for arg in vars(args):
        if hasattr(config, arg):
           setattr(config, arg, getattr(args, arg))
    return config

def trim_input_ids(input_ids: torch.tensor, pad_token_id, mask_token_id, num_masks: int):
    """
    Trim a sequence of input ids by removing all padding tokens and keeping at most a specific number of mask tokens.

    :param input_ids: the sequence of input token ids
    :param pad_token_id: the id of the pad token
    :param mask_token_id: the id of the mask tokens
    :param num_masks: the number of masks to keeps
    :return: the trimmed sequence of input ids
    """
    assert input_ids.shape[0] == 1
    input_ids_without_pad = [x for x in input_ids[0] if x != pad_token_id]

    trimmed_input_ids = []
    mask_count = 0
    for input_id in input_ids_without_pad:
        if input_id == mask_token_id:
            if mask_count >= num_masks:
                continue
            mask_count += 1
        trimmed_input_ids.append(input_id)

    return torch.tensor([trimmed_input_ids], dtype=torch.long, device=input_ids.device)


def get_aggregation(aggregation_type):
    if aggregation_type == "min":
        return torch.min
    elif aggregation_type == "max": 
        return torch.max
    elif aggregation_type == "mean":
        return torch.mean  
    elif aggregation_type == "sum":
        return torch.sum 
    
