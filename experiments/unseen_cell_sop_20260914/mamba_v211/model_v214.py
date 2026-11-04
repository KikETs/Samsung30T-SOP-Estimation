"""V214 = the V211 model with two more inputs: the causal ridge SOH estimate ((soh - 0.85) / 0.1, 0 when none yet) and a
flag that an estimate exists (PROTOCOL_MAMBA_V214.md). Everything else is model_v211 unchanged."""
import torch
from torch import nn
import model_v211 as B

D_MODEL, D_INNER, HEADS, STATE, LAYERS, CHUNK = B.D_MODEL, B.D_INNER, B.HEADS, B.STATE, B.LAYERS, B.CHUNK
PARAM_NAMES, response = B.PARAM_NAMES, B.response


class Model(B.Model):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(B.N_FEAT + 2, B.D_MODEL)

    def encode(self, V, I, T, dt, soh, has, sequential=False):
        f = torch.cat([B.features(V, I, T, dt), torch.stack([torch.where(has, (soh - .85) / .1, torch.zeros_like(soh)), has.float()], -1)], -1)
        u = self.embed(f)
        for blk in self.blocks: u, _ = blk(u, dt, sequential=sequential)
        return self.norm(u)
