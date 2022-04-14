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
