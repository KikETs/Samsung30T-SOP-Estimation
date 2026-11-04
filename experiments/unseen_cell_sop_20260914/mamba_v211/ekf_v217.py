"""V217 step 1: causal EKF SOC series for every stream (PROTOCOL_MAMBA_V217.md).

Filter: the paper's 2RC EKF (analysis/ekf_soc.py, class EKF, imported unchanged) with the adopted
configuration of repro/run_soc_loco.py: gamma 0, i_gate 1 A, rest_hold_s 30 s, R_volt = interp(SOH, [.70 .90 1.00],
[.110 .035 .015]). SOC on the paper's rated axis (1.0 at full, divided by 3.0 Ah).
Surfaces: ecm_pool.surfaces(h) for every stream of the fold with held-out cell h (built from the other five cells, so no
stream of the fold reads the held-out cell's characterisation); 'ALL' for the all-cell fit and for RPCWBY.
Restarts: the filter restarts at the first sample after every full charge (data_v211 rule: >= 600 s, reaches 4.19 V; its
end = last charging sample of the run) with SOC 1.0, V1 = V2 = 0 and the paper's initial covariance. From the charge start
to its end, and before the first full charge of a stream, there is no estimate (has = False). At a segment start inside a
run (logging, hidden or file gap) V1 = V2 = 0, the covariance returns to its initial value and the rest timer restarts.
SOH given to the filter: the causal ridge estimate (V214 series, median of the last five 1C runs) available at the charge
end, fixed until the next restart; 1.0 when no estimate exists yet. Missing temperatures: last finite value, else 25 C.
Nothing here reads a SOC or SOH label."""
import multiprocessing as mp, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import data_v211 as D

ANALYSIS_DIR = str(Path(__file__).resolve().parents[3] / 'analysis')
if ANALYSIS_DIR not in sys.path: sys.path.insert(0, ANALYSIS_DIR)
from ekf_soc import EKF  # noqa: E402

GATE, HOLD = 1.0, 30.0
_SURF = {}


def rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def surfaces(pool):
    if pool not in _SURF:
        import os, ecm_pool
        cwd = os.getcwd(); os.chdir(ANALYSIS_DIR)
        try: _SURF[pool] = ecm_pool.surfaces(pool)
        finally: os.chdir(cwd)
    return _SURF[pool]


def fill_T(T):
    T = np.asarray(T, np.float64).copy(); ok = np.isfinite(T)
    if not ok.any(): return np.full(len(T), 25.)
    idx = np.where(ok, np.arange(len(T)), 0); np.maximum.accumulate(idx, out=idx)
    T = T[idx]; T[:np.argmax(ok)] = 25.; return T


def charge_runs(s):
    """(start, end) of every full charge: same run and break rule as data_v211.full_charge_starts; end = last I > 0.1 A sample."""
    t, I, V, seg = s['t'], s['I'], s['V'], s['seg']
    idx = np.flatnonzero(I > .1)
    if not len(idx): return np.zeros(0, int), np.zeros(0, int)
    brk = np.r_[True, ((np.diff(idx) > 1) & (np.diff(t[idx]) >= 30.)) | (seg[idx][1:] != seg[idx][:-1])]
    run = np.cumsum(brk) - 1; starts = idx[brk]; ends = np.r_[idx[np.flatnonzero(brk)[1:] - 1], idx[-1]]
    vmax = np.full(run[-1] + 1, -np.inf); np.maximum.at(vmax, run, V[idx])
    full = (vmax >= 4.19) & (t[ends] - t[starts] >= 600.)
    return starts[full], ends[full]


def run_segment(s, a, e, soh, pool):
    """EKF over samples [a, e] from a fresh restart at a (SOC 1.0). Returns float32 SOC estimates."""
    sd, sc = surfaces(pool)
    f = EKF(sd, sc, float(soh), R_volt=rvolt(soh), gamma=0.0, i_gate=GATE, rest_hold_s=HOLD)
    p0 = f.P.copy(); f.x = np.zeros(f.n); f.x[0] = 1.0
    t, I, V, seg = s['t'], s['I'], s['V'], s['seg']; T = s['Tf']
    out = np.empty(e - a + 1, np.float32)
    for k in range(a, e + 1):
        if k > a and seg[k] != seg[k - 1]:
            f.x[1] = f.x[2] = 0.; f.P = p0.copy(); f.rest_s = 0.
        dt = 1.0 if k == a or seg[k] != seg[k - 1] else float(t[k] - t[k - 1])
        out[k - a] = f.step(float(I[k]), float(V[k]), float(T[k]), dt)[0]
    return out


def plan(s, soh, has):
    """Restart plan of a whole stream: list of (a, e, soh given to the filter) plus the charge runs."""
    cs, ce = charge_runs(s); n = len(s['t']); jobs = []
    for k, (c, e) in enumerate(zip(cs, ce)):
        a = int(e) + 1; b = int(cs[k + 1]) - 1 if k + 1 < len(cs) else n - 1
        if a <= b: jobs.append((a, b, float(soh[e]) if has[e] else 1.0))
    return jobs, cs, ce


def _job(args):
    name, a, e, soh, pool = args
    return a, run_segment(_STREAMS[name], a, e, soh, pool)


_STREAMS = {}


def series(streams, soh_series, pool, workers=6):
    """streams: {name: base series}; soh_series: {name: (soh, has)}. Returns {name: (soc, has_soc, soh_ekf)}. One job = one
    restart run, longest first. Six workers: the machine is shared with a running Quantum ESPRESSO job."""
    for nm, s in streams.items(): s['Tf'] = fill_T(s['T']); _STREAMS[nm] = s
    jobs = []; meta = {}
    for nm, s in streams.items():
        pj, cs, ce = plan(s, *soh_series[nm]); meta[nm] = (pj, cs, ce)
        jobs += [(nm, a, e, soh, pool) for a, e, soh in pj]
    jobs.sort(key=lambda j: -(j[2] - j[1]))
    res = {}
    # fork: the workers inherit the streams instead of receiving a pickled copy each
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork'), initializer=surfaces, initargs=(pool,)) as ex:
        for (nm, *_), (a, soc) in zip(jobs, ex.map(_job, jobs, chunksize=1)): res[(nm, a)] = soc
    out = {}
    for nm, s in streams.items():
        n = len(s['t']); soc = np.zeros(n, np.float32); hs = np.zeros(n, bool); se = np.zeros(n, np.float32)
        for a, e, soh in meta[nm][0]: soc[a:e + 1] = res[(nm, a)]; hs[a:e + 1] = True; se[a:e + 1] = soh
        out[nm] = (soc, hs, se)
    return out



def mendeley_order():
    """Permutation data_v213.base('mendeley') applies to the concatenated campaign files."""
    return np.argsort(np.concatenate([D.load(n)['t'] for n in D.MENDELEY]), kind='stable')


def soh_inputs(held):
    """V214 causal ridge series per stream name of data_v213 ('mendeley' = the time-ordered campaign files)."""
    z = np.load(D.HERE / 'runs' / f'v214_soh_{held}.npz'); out = {n: (z[f'{n}__soh'], z[f'{n}__has']) for n in D.UYPYDJ}
    o = mendeley_order()
    out['mendeley'] = tuple(np.concatenate([z[f'{n}__{k}'] for n in D.MENDELEY])[o] for k in ('soh', 'has'))
    return out


def main():
    """usage: python ekf_v217.py <held-out cell | NONE>; pool = the held-out cell, or ALL for NONE."""
    import json, time
    import data_v213 as S
    held = sys.argv[1]; pool = 'ALL' if held == 'NONE' else held
    out = D.HERE / 'runs' / f'v217_ekf_{held}.npz'; assert not out.exists()
    names = D.UYPYDJ + ['mendeley']; t0 = time.time()
    streams = {n: S.base(n) for n in names}; sohs = soh_inputs(held)
    for n in names: assert len(sohs[n][0]) == len(streams[n]['t']), n
    res = series(streams, sohs, pool); arrays = {}; rep = dict(held_out=held, pool=pool, minutes=(time.time() - t0) / 60, streams={})
    for n in names:
        soc, hs, se = res[n]; s = streams[n]
        arrays.update({f'{n}__soc': soc, f'{n}__has': hs, f'{n}__soh_ekf': se, f'{n}__soh_in': sohs[n][0], f'{n}__has_soh': sohs[n][1]})
        # label-free consistency check: coulomb count from the same restart (1.0), no voltage correction
        jobs, cs, ce = plan(s, *sohs[n]); d = []
        for a, e, _ in jobs:
            t, I = s['t'][a:e + 1], s['I'][a:e + 1]; dtk = np.r_[0., np.diff(t)]; dtk[np.r_[True, s['seg'][a + 1:e + 1] != s['seg'][a:e]]] = 0.
            d.append(soc[a:e + 1] - (1. + np.cumsum(I * dtk) / 3600. / 3.))
        d = np.abs(np.concatenate(d)) if d else np.zeros(1)
        rep['streams'][n] = dict(samples=int(len(soc)), full_charges=int(len(cs)), restarts=int(len(jobs)), has_fraction=float(hs.mean()),
                                 filter_soh_default_1_fraction=float(np.mean([not sohs[n][1][e] for e in ce])) if len(ce) else None,
                                 abs_diff_to_coulomb_count_median=float(np.median(d)), abs_diff_to_coulomb_count_q95=float(np.quantile(d, .95)),
                                 soc_range=[float(soc[hs].min()), float(soc[hs].max())] if hs.any() else None)
    np.savez(out, **arrays); json.dump(rep, open(out.with_suffix('.json'), 'w'), indent=1); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
