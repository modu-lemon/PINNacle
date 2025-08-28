import torch
import torch.nn as nn
import torch.nn.functional as F

from deepxde import config
from deepxde import nn as dde_nn

from src.model.fnn import FNN as LocalFNN
from src.model.laaf import DNN_LAAF, DNN_GAAF


class GatingNetwork(nn.Module):

    def __init__(self, input_dim, num_experts, hidden_layers=None, activation="tanh"):
        super().__init__()
        hidden_layers = hidden_layers or [64, 64]
        layers = []
        prev = input_dim
        act = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }.get(activation, nn.Tanh())
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h, dtype=config.real(torch)))
            layers.append(act)
            prev = h
        layers.append(nn.Linear(prev, num_experts, dtype=config.real(torch)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class MoE(nn.Module):
    """
    Mixture-of-Experts wrapper for PINN experts.

    All experts must accept input of shape (batch, input_dim) and return (batch, output_dim).
    """

    def __init__(self, experts, input_dim, output_dim, gating_hidden=None, gating_activation="tanh", top_k=0):
        super().__init__()
        assert len(experts) >= 1, "At least one expert required"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.top_k = top_k if top_k is not None else 0
        self.gating = GatingNetwork(input_dim, self.num_experts, hidden_layers=gating_hidden, activation=gating_activation)

        # Forward attribute passthrough for DeepXDE transforms if present
        self._input_transform = None
        self._output_transform = None
        self.regularizer = None

    def forward(self, inputs):
        x = inputs
        if getattr(self, "_input_transform", None) is not None:
            x = self._input_transform(x)

        weights = self.gating(x)  # (B, E)
        if self.top_k and self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(weights, k=self.top_k, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

        y_list = [expert(inputs) for expert in self.experts]  # each (B, D)
        y_stack = torch.stack(y_list, dim=-1)  # (B, D, E)
        y = (y_stack * weights.unsqueeze(1)).sum(dim=-1)  # (B, D)

        if getattr(self, "_output_transform", None) is not None:
            y = self._output_transform(inputs, y)
        return y


def build_expert(expert_type, input_dim, output_dim, hidden_layers):
    """
    Factory to build an expert network from shorthand.

    expert_type: "fnn", "laaf", "gaaf", "pfnn"
    hidden_layers: list[int]
    """
    t = expert_type.lower()
    if t == "fnn":
        # Prefer DeepXDE FNN to benefit from transforms
        try:
            return dde_nn.FNN([input_dim] + list(hidden_layers) + [output_dim], "tanh", "Glorot normal")
        except Exception:
            return LocalFNN([input_dim] + list(hidden_layers) + [output_dim], "tanh", "Glorot normal")
    if t == "laaf":
        assert len(hidden_layers) >= 1, "hidden_layers must be non-empty for LAAF"
        return DNN_LAAF(len(hidden_layers) - 1, hidden_layers[0], x_dim=input_dim, u_dim=output_dim)
    if t == "gaaf":
        assert len(hidden_layers) >= 1, "hidden_layers must be non-empty for GAAF"
        return DNN_GAAF(len(hidden_layers) - 1, hidden_layers[0], x_dim=input_dim, u_dim=output_dim)
    if t == "pfnn":
        # Use parallel FNN to allow per-output subnets; fall back to shared layers
        # Here we keep it simple: just use standard FNN when split mask not provided
        return dde_nn.FNN([input_dim] + list(hidden_layers) + [output_dim], "tanh", "Glorot normal")
    raise ValueError(f"Unknown expert_type: {expert_type}")


def build_moe(expert_types, pde, hidden_layers, gating_hidden=None, gating_activation="tanh", top_k=0):
    experts = [build_expert(t, pde.input_dim, pde.output_dim, hidden_layers) for t in expert_types]
    moe = MoE(experts, pde.input_dim, pde.output_dim, gating_hidden=gating_hidden, gating_activation=gating_activation, top_k=top_k)
    return moe

