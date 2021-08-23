import torch

from torch.nn import Module, functional as F
from typing import List

class SplitDropout(Module):
    def __init__(self, chunk_sizes: List[int], dropout_rates: List[float]):
        assert len(chunk_sizes) == len(dropout_rates)
        super().__init__()
        self.chunk_sizes = chunk_sizes
        self.dropout_rates = dropout_rates
    
    def forward(self, xs):
        if not self.training:
            return xs
        parts = torch.split(xs, self.chunk_sizes, dim=-1)
        result = []
        for slice, dropout in zip(parts, self.dropout_rates):
            result.append(F.dropout(slice, dropout, self.training))
        return torch.cat(result, dim=-1)
