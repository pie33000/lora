import math

import torch
import torch.nn as nn


class LoRaLayer(nn.Module):
    def __init__(self, wrapped_layer, rank=4, alpha=1.0):
        super(LoRaLayer, self).__init__()
        self.wrapped_layer = wrapped_layer
        if isinstance(self.wrapped_layer, nn.Linear):
            in_features = wrapped_layer.in_features
            out_features = wrapped_layer.out_features
        else:
            raise ValueError("Unsupported layer type")

        self.rank = rank
        self.scaling_factor = alpha / rank

        # Define low-rank matrice

        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))

        # Initialize the low-rank matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Calculate the low-rank adaptation
        lora_adjustment = torch.matmul(x, self.lora_A).matmul(self.lora_B)

        # Add the low-rank adjustment to the output of the wrapped layer
        return self.wrapped_layer(x) + self.scaling_factor * lora_adjustment
