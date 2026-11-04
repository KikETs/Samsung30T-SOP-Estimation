"""V213 lifelong-stateful variant of the V211 model (PROTOCOL_MAMBA_V213.md).

Changes against model_v211 (everything else, including the circuit head and response, is imported unchanged):
  * 8 heads x 4 channels (d_inner 32 as before), initial step rates log-spaced 1 .. 1e-7 per second, so the slowest heads
    remember for months instead of about 3 h.
  * Gap step. A sample flagged `gap` starts after a stretch that must not be read (a same-search blackout on RPCWBY, a
    logging gap, or a file gap). Its decay uses the gap length and a learned per-head gap rate softplus(gap_bias) that
    does not depend on the input; its own measurement is injected with a nominal 1 s step. Nothing inside the gap is
    read. gap_bias starts equal to dt_bias, i.e. by default a gap decays like a rest of the same length.
  * encode() takes and returns the per-layer state, so a stream can be processed chunk by chunk with the state carried.
"""
import torch
import torch.nn.functional as F
from torch import nn
import model_v211 as B

D_MODEL, D_INNER, HEADS, STATE, LAYERS, CHUNK = 16, 32, 8, 8, 2, 64
HEAD_DIM = D_INNER // HEADS
PARAM_NAMES, sat, response, features = B.PARAM_NAMES, B.sat, B.response, B.features


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = B.RMSNorm(D_MODEL)
        self.in_proj = nn.Linear(D_MODEL, 2 * D_INNER + 2 * STATE + HEADS, bias=False)
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)
        self.A_log = nn.Parameter(torch.zeros(HEADS))
        rate = torch.logspace(0, -7, HEADS)
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(rate)))
        self.gap_bias = nn.Parameter(torch.log(torch.expm1(rate)))
        self.D = nn.Parameter(torch.ones(HEADS))
        self.out_norm = B.RMSNorm(D_INNER)

    def forward(self, u, dt, gap, h0=None, sequential=False):
        z, x, B_, C, d = torch.split(self.in_proj(self.norm(u)), [D_INNER, D_INNER, STATE, STATE, HEADS], -1)
        rate = F.softplus(d + self.dt_bias)                                          # b l h
        g = gap[..., None]
        inject = torch.where(g, rate, dt[..., None] * rate)                          # gap sample: nominal 1 s injection
        decay = torch.where(g, dt[..., None] * F.softplus(self.gap_bias), dt[..., None] * rate)
        dA = -decay * torch.exp(self.A_log)
        x = x.reshape(*x.shape[:-1], HEADS, HEAD_DIM)
        fn = B.ssd_sequential if sequential else ssd
        y, h = fn(x * inject[..., None], dA, B_, C, h0)
        y = y + self.D[:, None] * x
        y = self.out_norm(y.reshape(*y.shape[:-2], D_INNER)) * F.silu(z)
        return u + self.out_proj(y), h


def ssd(X, dA, B_, C, h0=None):
    """model_v211.ssd with this module's CHUNK (identical algorithm)."""
    b, l, h, p = X.shape; c = l // CHUNK
    X = X.reshape(b, c, CHUNK, h, p); B_ = B_.reshape(b, c, CHUNK, -1); C = C.reshape(b, c, CHUNK, -1)
    A = dA.reshape(b, c, CHUNK, h).permute(0, 3, 1, 2); Acs = torch.cumsum(A, -1)
    Y = torch.einsum('bcln,bcsn,bhcls,bcshp->bclhp', C, B_, torch.exp(B.segsum(A)), X)
    states = torch.einsum('bcln,bhcl,bclhp->bchpn', B_, torch.exp(Acs[..., -1:] - Acs), X)
    h0 = torch.zeros_like(states[:, :1]) if h0 is None else h0[:, None]
    states = torch.cat([h0, states], 1)
    new = torch.einsum('bhzc,bchpn->bzhpn', torch.exp(B.segsum(F.pad(Acs[..., -1], (1, 0)))), states)
    Y = Y + torch.einsum('bcln,bchpn,bhcl->bclhp', C, new[:, :-1], torch.exp(Acs))
    return Y.reshape(b, l, h, p), new[:, -1]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(B.N_FEAT, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = B.RMSNorm(D_MODEL)
        ref = B.Model()                                      # same head and head initialisation as V211
        self.head = nn.Linear(D_MODEL, 2 * len(PARAM_NAMES)); self.head.load_state_dict(ref.head.state_dict())
        self.params = ref.params.__func__.__get__(self)

    def encode(self, V, I, T, dt, gap, h0=None, sequential=False):
        """(b, l) inputs, l a multiple of CHUNK unless sequential; h0: list of per-layer states or None.
        Returns the encoder output (b, l, D_MODEL) and the list of final per-layer states."""
        u = self.embed(features(V, I, T, dt)); hs = []
        for k, blk in enumerate(self.blocks):
            u, h = blk(u, dt, gap, None if h0 is None else h0[k], sequential); hs.append(h)
        return self.norm(u), hs
