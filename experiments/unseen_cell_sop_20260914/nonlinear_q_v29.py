"""Causal two-RC voltage model with a monotone signed-square charge term."""
import torch
from torch import nn
from history_ecm_2rc_v1 import Encoder as BaseEncoder, response as base_response

B_MAX = 0.00005


class Encoder(BaseEncoder):
    def __init__(self, n_features):
        super().__init__(n_features)
        state = torch.random.get_rng_state()
        self.charge_head = nn.Linear(32, 1)
        nn.init.zeros_(self.charge_head.weight)
        with torch.no_grad():
            self.charge_head.bias.fill_(float(torch.logit(torch.tensor(1e-7 / B_MAX))))
        torch.random.set_rng_state(state)

    def forward(self, x, history_time, history_current):
        p = super().forward(x, history_time, history_current)
        b = B_MAX * torch.sigmoid(self.charge_head(self.net[:-1](x)))
        return torch.cat([p, b], dim=1)


def response(parameters, times, currents, query_times, query_currents):
    lo = times[:, :-1, None]
    hi = times[:, 1:, None]
    dt = (torch.minimum(hi, query_times[:, None, :]) - lo).clamp_min(0)
    q = (currents[:, :-1, None] * dt).sum(1)
    return (base_response(parameters[:, :8], times, currents, query_times, query_currents)
            + parameters[:, 8, None] * q * q.abs())
