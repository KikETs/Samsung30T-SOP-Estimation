"""V211 data rules shared by training and every evaluation (PROTOCOL_MAMBA_V211.md).

Series: t (s, absolute), V, I (charge > 0), T, segment. Mendeley sub-second logging is merged into 1 s bins (I and T
averaged, V taken at the bin end, t = last sample time); other sources keep their native samples.
Full-charge start: first sample of a charging run (I > 0.1 A, interruptions shorter than 30 s merged) that lasts at least
600 s and whose voltage reaches >= 4.19 V. History of a query at sample q: from the latest full-charge start within 6 h before q, otherwise from
max(t_q - 6 h, segment start), up to and including q.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
UYPYDJ = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
MENDELEY = ['mendeley_-20degC', 'mendeley_-10degC', 'mendeley_0degC', 'mendeley_10degC', 'mendeley_25degC', 'mendeley_40degC']
MAX_HIST_S, MIN_HIST_S, WINDOW_S, MAX_WIN = 6 * 3600., 600., 30., 48


def bin_1s(t, V, I, T, seg):
    """Merge runs of samples closer than 1 s: consecutive samples go to the same bin while the bin spans < 1 s."""
    n = len(t); start = np.zeros(n, bool); start[0] = True
    anchor = t[0]
    for k in range(1, n):  # sequential by construction; only called on Mendeley (a few million samples)
        if seg[k] != seg[k - 1] or t[k] - anchor >= 1.0 - 1e-9: start[k] = True; anchor = t[k]
    b = np.cumsum(start) - 1; nb = b[-1] + 1
    cnt = np.bincount(b, minlength=nb).astype(np.float64)
    last = np.r_[np.flatnonzero(start)[1:] - 1, n - 1]
    Im = np.bincount(b, weights=I.astype(np.float64), minlength=nb) / cnt
    Tf = np.where(np.isfinite(T), T, 0.).astype(np.float64); Tc = np.bincount(b, weights=np.isfinite(T).astype(float), minlength=nb)
    Tm = np.where(Tc > 0, np.bincount(b, weights=Tf, minlength=nb) / np.maximum(Tc, 1), np.nan)
    return t[last], V[last], Im.astype(np.float32), Tm.astype(np.float32), seg[last]


def clean(t, V, I, T, seg):
    """Logging glitches (0-3 samples per cell, next to gaps): drop |I| > 40 A or V outside 2.40-4.30 V; T outside -30..60 C -> NaN."""
    ok = (np.abs(I) <= 40.) & (V >= 2.40) & (V <= 4.30)
    T = np.where((T >= -30.) & (T <= 60.), T, np.nan).astype(np.float32)
    return t[ok], V[ok], I[ok], T[ok], seg[ok], ok


def load(name):
    with np.load(CACHE / f'{name}.npz') as z:
        t, V, I, T, seg = z['t'], z['V'], z['I'], z['T'], z['segment']
    n_raw = len(t); t, V, I, T, seg, ok = clean(t, V, I, T, seg)
    raw_index = np.flatnonzero(ok)  # cache index of every kept sample: bench rows point into the uncleaned cache
    if name.startswith('mendeley'): t, V, I, T, seg = bin_1s(t, V, I, T, seg); raw_index = None
    dt = np.r_[1., np.diff(t)]; dt[np.r_[True, seg[1:] != seg[:-1]]] = 1.  # first sample of a segment: nominal 1 s
    return dict(name=name, t=t, V=V.astype(np.float32), I=I.astype(np.float32), T=T.astype(np.float32), seg=seg, dt=dt.astype(np.float32),
                raw_index=raw_index)


def full_charge_starts(s):
    """Indices of full-charge starts (see module docstring)."""
    t, I, V, seg = s['t'], s['I'], s['V'], s['seg']
    on = I > .1; idx = np.flatnonzero(on)
    if not len(idx): return np.array([], int)
    # a break needs non-charging samples in between lasting >= 30 s; sparse logging inside one charge is not a break
    brk = np.r_[True, ((np.diff(idx) > 1) & (np.diff(t[idx]) >= 30.)) | (seg[idx][1:] != seg[idx][:-1])]
    run = np.cumsum(brk) - 1; starts = idx[brk]
    vmax = np.full(run[-1] + 1, -np.inf); np.maximum.at(vmax, run, V[idx])
    tend = np.full(run[-1] + 1, -np.inf); np.maximum.at(tend, run, t[idx])
    # >= 600 s of charging: HPPC charge pulses near the top also touch 4.19 V but are not a recharge (smoke test)
    return starts[(vmax >= 4.19) & (tend - t[starts] >= 600.)]


def history_start(s, q, charge_starts):
    """Start index of the history of query sample q."""
    t, seg = s['t'], s['seg']
    seg0 = np.searchsorted(seg, seg[q], 'left')
    lo = max(np.searchsorted(t, t[q] - MAX_HIST_S, 'left'), seg0)
    k = np.searchsorted(charge_starts, q, 'right') - 1
    if k >= 0 and charge_starts[k] >= lo: return int(charge_starts[k])
    return int(lo)
