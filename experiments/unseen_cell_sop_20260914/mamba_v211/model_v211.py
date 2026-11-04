"""V211 tiny selective state-space encoder (Mamba-2 / SSD form) with an equivalent-circuit head.

Encoder input per sample: V, I, T and the sample interval dt (irregular logging is handled by making the step size
Delta = dt * softplus(.)), no SOC / SOH / capacity / cell id. Two layers, d_model 16, d_inner 32, 4 heads x 8 channels,
state 8, one scalar decay per head (so a microcontroller step needs 4 exponentials, not d_inner * N).

Head: at a query sample t0 the last encoder output gives, per pulse direction, the circuit parameters
R0, R1, tau1, R2, tau2 (V57 ranges), k (linear charge term), b (signed-square charge term), Rsat (saturating ohmic term)
and the two branch overpotentials u1(0), u2(0) at t0. The voltage change after t0 for any current profile is then

  dV(tau) = R0 (I(tau) - I0) + sum_k [u_k(tau) - u_k(0)] + k q + b q|q| + Rsat (sat(I(tau)) - sat(I0)),
  u_k(tau_j) = u_k(tau_{j-1}) e^{-d/tau_k} + R_k I_j (1 - e^{-d/tau_k}),   q = integral of I,

with I_j held on (tau_{j-1}, tau_j]. Charge current is positive (raw-data convention), so resistances stay positive in
both directions. The chunked scan (training) and the per-sample recurrence (reference / MCU) are the same model.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

D_MODEL, D_INNER, HEADS, STATE, LAYERS, CHUNK = 16, 32, 4, 8, 2, 64
HEAD_DIM = D_INNER // HEADS
N_FEAT = 4


def features(V, I, T, dt):
    """Fixed input scaling. T may be NaN (missing channel) -> 25 C."""
    T = torch.where(torch.isfinite(T), T, torch.full_like(T, 25.))
    return torch.stack([(V - 3.7) / .5, I / 10., (T - 25.) / 20., torch.log(dt.clamp_min(1e-3)) / 5.], -1)


def segsum(x):
    """x (..., L) -> (..., L, L) with out[i, j] = sum_{j < k <= i} x_k for j <= i, -inf above the diagonal."""
    L = x.size(-1)
    x = x[..., None].expand(*x.shape, L)
    x = x.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device), -1), 0.)
    s = torch.cumsum(x, dim=-2)
    return s.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device), 0), -math.inf)


def ssd(X, dA, B, C, h0=None):
    """Chunked SSD scan. X (b,l,h,p) already scaled by Delta; dA (b,l,h) <= 0; B, C (b,l,n); l multiple of CHUNK.
    Returns Y (b,l,h,p) and the final state (b,h,p,n)."""
    b, l, h, p = X.shape; c = l // CHUNK
    X = X.reshape(b, c, CHUNK, h, p); B = B.reshape(b, c, CHUNK, -1); C = C.reshape(b, c, CHUNK, -1)
    A = dA.reshape(b, c, CHUNK, h).permute(0, 3, 1, 2)                     # b h c l
    Acs = torch.cumsum(A, -1)
    Ld = torch.exp(segsum(A))                                              # b h c l s
    Y = torch.einsum('bcln,bcsn,bhcls,bcshp->bclhp', C, B, Ld, X)
    states = torch.einsum('bcln,bhcl,bclhp->bchpn', B, torch.exp(Acs[..., -1:] - Acs), X)
    if h0 is None: h0 = torch.zeros_like(states[:, :1])
    else: h0 = h0[:, None]
    states = torch.cat([h0, states], 1)
    dc = torch.exp(segsum(F.pad(Acs[..., -1], (1, 0))))                   # b h c+1 c+1
    new = torch.einsum('bhzc,bchpn->bzhpn', dc, states)
    Y = Y + torch.einsum('bcln,bchpn,bhcl->bclhp', C, new[:, :-1], torch.exp(Acs))
    return Y.reshape(b, l, h, p), new[:, -1]


def ssd_sequential(X, dA, B, C, h0=None):
    """Per-sample recurrence h_t = exp(dA_t) h_{t-1} + X_t (x) B_t, y_t = C_t . h_t (reference and MCU form)."""
    b, l, h, p = X.shape
    s = torch.zeros(b, h, p, B.shape[-1], dtype=X.dtype, device=X.device) if h0 is None else h0.clone()
    ys = []
    for t in range(l):
        s = torch.exp(dA[:, t])[..., None, None] * s + X[:, t, :, :, None] * B[:, t, None, None, :]
        ys.append(torch.einsum('bhpn,bn->bhp', s, C[:, t]))
    return torch.stack(ys, 1), s


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.in_proj = nn.Linear(D_MODEL, 2 * D_INNER + 2 * STATE + HEADS, bias=False)
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)
        self.A_log = nn.Parameter(torch.zeros(HEADS))                      # |A| = 1 at start
        # per-head step rate softplus(bias) log-spaced 1e-4 .. 1 per second: memories from ~1 s to ~3 h
        rate = torch.logspace(0, -4, HEADS)
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(rate)))
        self.D = nn.Parameter(torch.ones(HEADS))
        self.out_norm = RMSNorm(D_INNER)

    def split(self, u, dt):
        z, x, B, C, d = torch.split(self.in_proj(self.norm(u)), [D_INNER, D_INNER, STATE, STATE, HEADS], -1)
        delta = dt[..., None] * F.softplus(d + self.dt_bias)                   # b l h, zero where dt = 0 (padding)
        dA = -delta * torch.exp(self.A_log)
        x = x.reshape(*x.shape[:-1], HEADS, HEAD_DIM)
        return z, x, delta, dA, B, C

    def finish(self, u, z, x, y):
        y = y + self.D[:, None] * x
        y = self.out_norm(y.reshape(*y.shape[:-2], D_INNER)) * F.silu(z)
        return u + self.out_proj(y)

    def forward(self, u, dt, h0=None, sequential=False):
        z, x, delta, dA, B, C = self.split(u, dt)
        fn = ssd_sequential if sequential else ssd
        y, h = fn(x * delta[..., None], dA, B, C, h0)
        return self.finish(u, z, x, y), h


PARAM_NAMES = ['R0', 'R1', 'tau1', 'R2', 'tau2', 'k', 'b', 'Rsat', 'u1', 'u2']


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(N_FEAT, D_MODEL)
        self.blocks = nn.ModuleList([Block(i) for i in range(LAYERS)])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 2 * len(PARAM_NAMES))                   # [discharge | charge]
        nn.init.zeros_(self.head.weight)
        init = torch.tensor([(.015 - .0001) / .1999, (.005 - .00005) / .09995, (2. - .05) / 59.9, (.005 - .00005) / .09995,
                             (40. - 2.) / 58., 1e-6 / 5e-3 + 1e-4, 1e-7 / 5e-5 + 1e-4, 1e-4 / .1 + 1e-4, .5, .5])
        with torch.no_grad(): self.head.bias.copy_(torch.logit(init.clamp(1e-4, 1 - 1e-4)).repeat(2))

    def encode(self, V, I, T, dt, sequential=False):
        """Inputs (b, l) with l a multiple of CHUNK unless sequential. Returns the encoder output (b, l, D_MODEL)."""
        u = self.embed(features(V, I, T, dt))
        for blk in self.blocks: u, _ = blk(u, dt, sequential=sequential)
        return self.norm(u)

    def params(self, y, charge):
        """y (n, D_MODEL), charge (n,) bool -> dict of (n,) tensors with the V57-style bounded ranges."""
        raw = self.head(y).reshape(-1, 2, len(PARAM_NAMES))
        q = torch.sigmoid(raw[torch.arange(len(y), device=y.device), charge.long()])
        tau1 = .05 + 59.9 * q[:, 2]
        return dict(R0=.0001 + .1999 * q[:, 0], R1=.00005 + .09995 * q[:, 1], tau1=tau1, R2=.00005 + .09995 * q[:, 3],
                    tau2=tau1 + .0001 + (3600. - tau1 - .0001) * q[:, 4], k=5e-3 * q[:, 5], b=5e-5 * q[:, 6], Rsat=.1 * q[:, 7],
                    u1=.3 * (2 * q[:, 8] - 1), u2=.3 * (2 * q[:, 9] - 1))


def sat(I): return 10. * torch.tanh(I / 10.)


def response(p, tau, I, I0, mask):
    """Voltage change after t0. p: dict of (n,); tau (n, w) times after t0 (> 0, increasing where mask);
    I (n, w) current held on (tau_{j-1}, tau_j]; I0 (n,) current at t0; mask (n, w) valid samples. Returns (n, w)."""
    d = torch.diff(torch.cat([torch.zeros_like(tau[:, :1]), tau], 1), dim=1) * mask
    q = torch.cumsum(I * d, 1)
    out = p['R0'][:, None] * (I - I0[:, None]) + p['k'][:, None] * q + p['b'][:, None] * q * q.abs() \
        + p['Rsat'][:, None] * (sat(I) - sat(I0)[:, None])
    for R, tc, u0 in ((p['R1'], p['tau1'], p['u1']), (p['R2'], p['tau2'], p['u2'])):
        a = torch.exp(-d / tc[:, None]); u = u0.clone(); us = []
        for j in range(tau.shape[1]):
            u = u * a[:, j] + R * I[:, j] * (1 - a[:, j]); us.append(u)
        out = out + torch.stack(us, 1) - u0[:, None]
    return out * mask
