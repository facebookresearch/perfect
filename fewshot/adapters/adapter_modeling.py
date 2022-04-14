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