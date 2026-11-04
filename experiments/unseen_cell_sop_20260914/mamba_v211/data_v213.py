"""V213 lifelong streams (PROTOCOL_MAMBA_V213.md). Built on data_v211 (same cleaning, Mendeley 1 s bins, full-charge rule).

A stream is one cell's whole cleaned series in time order. Every segment start (logging gap, hidden gap, file gap) becomes
a gap sample: dt = the real gap length, gap = True, and the state is carried through it by the model's gap step instead of
being reset. The Mendeley cell's six temperature campaigns are one stream in time order.
Training blackouts: before a random 30 % of full-charge starts, the preceding 1-40 h are removed and the charge start
becomes a gap sample (the RPCWBY same-search blackout ends at the recharge start H0 in the same way)."""
import numpy as np
import data_v211 as D

WINDOW_S, MAX_WIN = D.WINDOW_S, D.MAX_WIN


def base(name):
    if name == 'mendeley':
        parts = [D.load(n) for n in D.MENDELEY]; off = 0; segs = []
        for p in parts: segs.append(p['seg'] + off); off = segs[-1].max() + 1
        order = np.argsort(np.concatenate([p['t'] for p in parts]), kind='stable')
        s = {k: np.concatenate([p[k] for p in parts])[order] for k in ('t', 'V', 'I', 'T')}
        s['seg'] = np.concatenate(segs)[order]; s['raw_index'] = None
        s['seg'] = np.r_[0, np.cumsum(s['seg'][1:] != s['seg'][:-1])]
    else:
        p = D.load(name); s = {k: p[k] for k in ('t', 'V', 'I', 'T', 'seg', 'raw_index')}
    s['name'] = name; s['cs'] = D.full_charge_starts(dict(t=s['t'], I=s['I'], V=s['V'], seg=s['seg']))
    return s


def stream(s, keep=None):
    """Arrays for the model from a base series and an optional keep mask (blackouts). gap = first kept sample of a segment
    or first kept sample after removed samples; dt = time since the previous kept sample (1 s for the very first)."""
    idx = np.arange(len(s['t'])) if keep is None else np.flatnonzero(keep)
    t = s['t'][idx]; seg = s['seg'][idx]
    jump = np.r_[False, (np.diff(idx) > 1) | (seg[1:] != seg[:-1])]
    dt = np.r_[1., np.diff(t)].astype(np.float32)
    dt[~jump & (np.arange(len(t)) > 0)] = np.maximum(dt[~jump & (np.arange(len(t)) > 0)], 1e-3)
    out = dict(idx=idx, t=t, V=s['V'][idx], I=s['I'][idx], T=s['T'][idx], dt=dt, gap=jump)
    # query support: last sample within (t0, t0 + 30 s] that is not a gap and not beyond the next gap
    n = len(t); nxt_gap = np.r_[np.flatnonzero(jump), n][np.searchsorted(np.flatnonzero(jump), np.arange(n), 'right')]
    wend = np.minimum(np.searchsorted(t, t + WINDOW_S, 'right') - 1, nxt_gap - 1)
    out['wend'] = wend
    hi = np.abs(out['I']) >= .5; cum = np.r_[0, np.cumsum(hi)]
    ok = wend >= np.arange(n) + 2
    out['active'] = ok & ((cum[wend + 1] - cum[np.arange(n) + 1]) > 0)
    out['onset'] = ok & np.r_[np.abs(np.diff(out['I'])) >= 1., False]
    return out


def blackout_keep(s, rng, p=.3, lo=3600., hi=40 * 3600.):
    keep = np.ones(len(s['t']), bool); t = s['t']; last_end = -np.inf
    for c in s['cs']:
        if rng.random() >= p: continue
        a = int(np.searchsorted(t, t[c] - rng.uniform(lo, hi), 'left'))
        if t[a] <= last_end or a >= c: continue
        keep[a:c] = False; last_end = t[c]
    return keep
