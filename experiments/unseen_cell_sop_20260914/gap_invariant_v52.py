"""Parameter inference is independent of planned gap; physics keeps actual time."""
import torch
from saturating_current_v39 import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    def __init__(self, n_features):
        if n_features != 74:
            raise ValueError('V52 requires the original 74 feature layout')
        super().__init__(n_features)
        # A legacy encoder must reject this checkpoint instead of silently
        # loading weights without the inference-time mask.
        self.register_buffer('_gap_invariant_v52', torch.tensor([73, 147]))

    def forward(self, x, history_time, history_current):
        if x.shape[1] != 148:
            raise ValueError('Expected 74 standardized values and 74 missing flags')
        state = x.clone()
        state[:, 73] = 0.
        state[:, 147] = 0.
        return super().forward(state, history_time, history_current)
