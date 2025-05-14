import numpy as np
import torch
from torch import nn


class Self_attention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(Self_attention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / np.sqrt(dim_k)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        q = self.linear_q(x)  # [batch_size, n, dim_k]
        k = self.linear_k(x)  # [batch_size, n, dim_k]
        v = self.linear_v(x)  # [batch_size, n, dim_v]
        dist = (q @ k.transpose(-1, -2)) * self._norm_fact  # [batch_size, n, n]
        dist = torch.softmax(dist, dim=-1)  # [batch_size, n, n]
        attention = self.dropout(dist) @ v
        return attention


class Add_norm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(Add_norm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.layer_norm(self.dropout(y) + x)
