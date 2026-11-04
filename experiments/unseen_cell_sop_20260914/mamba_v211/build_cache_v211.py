"""V211 step 1: full-cycle time-series cache for the sequence-model track (no model, no RPCWBY truth used).

Training sources (aging datasets only, user decision 2026-09-23):
  UYPYDJ  every weekly whole_Schedule .mat (six cells, 25 C chamber); files whose temperature channel failed
          (analysis/temp_audit_all.csv: bad_frac > 0 or T_med outside 15-45 C) are excluded whole.
  Mendeley every .mat of the fresh-cell campaign, six ambient temperatures, in file-date order.
Evaluation inputs (RPCWBY judged rows): raw samples from the admissible history start H0 (precheck/history_windows.csv)
  to the end of the query pulse. The model may only read samples before the query pulse; the pulse samples are kept
  for waveform diagnostics.

Stored per sample: absolute time t (float64 s), V, I, T (float32; I > 0 = charge, as in all three raw sources) and a
segment id. A new segment starts at a file boundary with a gap > 300 s, or inside a file at a gap > 300 s, or at a
gap > 120 s with |I| > 0.5 A on either side (logging loss while current flowed), or at a hidden gap: a jump > 30 s
(> 120 s for minute-resolution stamps) in wall clock minus the relative Time channel.
"""
import io, json, re, zipfile, datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'cache'
AUDIT = ROOT / 'analysis/temp_audit_all.csv'
MEND = ROOT / 'raw/Mendeley'
RPC = {'RPCWBY_TEST1': ROOT / 'raw/RPCWBY/1_Test_1.zip', 'RPCWBY_TEST2': ROOT / 'raw/RPCWBY/2_Test_2.zip',
       'RPCWBY_TEST3': ROOT / 'raw/RPCWBY/3_Test_3.zip'}
WIN = HERE / 'precheck/history_windows.csv'
KEYS = ROOT / 'experiments/same_search_exclusion_v157_20260921/keys/row_keys.csv'
GAP_S, GAP_ACTIVE_S, ACTIVE_A = 300., 120., .5  # 30 s cut every sample of 60 s-logged Mendeley charges (attempt04)


def parse_stamp(s):
    s = str(s).strip()
    for f in ('%Y-%m-%d %I:%M:%S %p', '%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d %I:%M %p',
              '%m/%d/%Y %I:%M %p'):
        try: return datetime.datetime.strptime(s, f).timestamp()
        except ValueError: pass
    raise ValueError(f'unparsed timestamp {s!r}')


def mat_struct(blob):
    m = sio.loadmat(io.BytesIO(blob), squeeze_me=True, struct_as_record=False)
    return m[next(k for k in m if not k.startswith('__'))]


def field(s, *names):
    for n in names:
        if n in s._fieldnames: return np.asarray(getattr(s, n), dtype=float).ravel()
    return None


def read_mat(job):
    """job = (source, zip or None, member or path) -> dict of arrays and a note."""
    src, zp, name = job
    blob = zipfile.ZipFile(zp).read(name) if zp else Path(name).read_bytes()
    s = mat_struct(blob)
    t = field(s, 'Time'); V = field(s, 'Voltage'); I = field(s, 'Current')
    T = field(s, 'Battery_temp_DegC', 'Battery_Temp_degC', 'Battery_Temp_DegC')
    if t is None or V is None or I is None or t.size < 2: return dict(src=src, name=name, skip='no series')
    if T is None or T.size != t.size: T = np.full_like(t, np.nan)
    stamp = np.asarray(getattr(s, 'TimeStamp')).ravel() if 'TimeStamp' in s._fieldnames else None
    ok = np.isfinite(t) & np.isfinite(V) & np.isfinite(I)
    if stamp is None or stamp.size != ok.size: return dict(src=src, name=name, skip='no wall-clock stamps')
    # The relative Time channel can silently skip logging gaps (e.g. 17.8 h hidden behind a 1 s step in CC week 2), so the
    # absolute time comes from wall_time(): forward jumps of wall clock minus Time are hidden gaps and hard breaks.
    w = (pd.to_datetime(pd.Series(stamp), format='mixed') - pd.Timestamp('1970-01-01')).dt.total_seconds().to_numpy()
    ok &= np.isfinite(w)
    t, V, I, T, w = t[ok], V[ok], I[ok], T[ok], w[ok]
    keep = np.r_[True, np.diff(t) > 0]  # drop duplicate / backward relative-clock samples
    t, V, I, T, w = t[keep], V[keep], I[keep], T[keep], w[keep]
    minute = bool(np.all(np.mod(w, 60.) == 0.)); thr = 120. if minute else 30.
    t_abs, hidden, n_back = wall_time(t, w, thr)
    # both clocks step back together where a file re-logs an earlier stretch (Mendeley 40 C Mixed3/Mixed4, attempt06):
    # keep the first record, drop replayed samples until time moves past the latest kept one, and break there
    mono = np.r_[True, t_abs[1:] > np.maximum.accumulate(t_abs)[:-1]]
    if not mono.all():
        lost = np.cumsum(~mono)[mono]
        t_abs, V, I, T = t_abs[mono], V[mono], I[mono], T[mono]
        hidden = hidden[mono] | np.r_[False, np.diff(lost) > 0]
    replayed = int((~mono).sum())
    # a piece of < 10 samples between hidden gaps is a stray record (BCNP1s week 10 ends with one sample 19 h later,
    # inside the next week's file: attempt01); drop it
    edges = np.r_[0, np.flatnonzero(hidden), len(t)]; stray = np.zeros(len(t), bool)
    for a, b in zip(edges[:-1], edges[1:]):
        if len(edges) > 2 and b - a < 10: stray[a:b] = True  # only between hidden gaps; tiny whole files stay (attempt02)
    if stray.any():
        lost = np.cumsum(stray)[~stray]  # stray samples removed so far, seen at each kept sample
        t_abs, V, I, T = t_abs[~stray], V[~stray], I[~stray], T[~stray]
        hidden = hidden[~stray] | np.r_[False, np.diff(lost) > 0]
    if len(t_abs) < 2: return dict(src=src, name=name, skip='fewer than 2 samples after cleaning')
    jump = np.flatnonzero(hidden)
    return dict(src=src, name=name, t=t_abs, V=V.astype(np.float32), I=I.astype(np.float32), T=T.astype(np.float32), hidden=hidden,
                abs_time=True, minute_stamps=minute, hidden_gaps=int(len(jump)), stray_dropped=int(stray.sum()),
                clock_set_back=int(n_back), replayed_dropped=replayed,
                n_dropped=int((~ok).sum() + (~keep).sum()))


def wall_time(t, w, thr):
    """Absolute time = relative clock + wall offset at the start + the sum of FORWARD offset jumps (> thr): a forward jump is
    a hidden logging gap (hard break); a backward jump is a clock set back (the relative clock keeps the true elapsed time),
    so it is ignored (attempt05 split pieces at set-backs too and broke on one). Offsets are per-piece medians."""
    off = w - t; d = np.diff(off); fwd = d > thr
    # per-piece median offset between forward jumps: averages out the 1 s (or 60 s) stamp rounding; a cumulative sum of
    # jump sizes carried that rounding forward and moved whole weeks by > 1.5 s against the HPPC files (attempt07)
    edges = np.r_[0, np.flatnonzero(fwd) + 1, len(t)]; t_abs = np.empty_like(t)
    for a, b in zip(edges[:-1], edges[1:]): t_abs[a:b] = t[a:b] + np.median(off[a:b])
    return t_abs, np.r_[False, fwd], int((d < -thr).sum())


def segments(t, I, file_break):
    dt = np.diff(t); a = np.abs(I)
    brk = (dt > GAP_S) | ((dt > GAP_ACTIVE_S) & ((a[:-1] > ACTIVE_A) | (a[1:] > ACTIVE_A))) | file_break[1:]
    return np.r_[0, np.cumsum(brk)].astype(np.int32), int(brk.sum())


def pack(name, parts, meta):
    parts = sorted(parts, key=lambda p: p['t'][0])
    for a, b in zip(parts[:-1], parts[1:]):  # trim the earlier file where two files still overlap in time
        cut = a['t'] >= b['t'][0]; a['trimmed'] = int(cut.sum())
        assert a['trimmed'] < 1000, (name, a['name'], a['trimmed'])
        for k in ('t', 'V', 'I', 'T', 'hidden'): a[k] = a[k][~cut]
    t = np.concatenate([p['t'] for p in parts]); V = np.concatenate([p['V'] for p in parts])
    I = np.concatenate([p['I'] for p in parts]); T = np.concatenate([p['T'] for p in parts])
    fid = np.concatenate([np.full(len(p['t']), k, np.int16) for k, p in enumerate(parts)])
    fb = (np.r_[False, np.diff(fid) != 0] & np.r_[False, np.diff(t) > GAP_S]) | np.concatenate([p['hidden'] for p in parts])
    back = np.r_[False, np.diff(t) <= 0]
    assert not back.any(), f'{name}: overlapping files'
    seg, nbrk = segments(t, I, fb)
    hid = np.concatenate([p['hidden'] for p in parts])
    np.savez(OUT / f'{name}.npz', t=t, V=V, I=I, T=T, file_id=fid, segment=seg, hidden=hid)
    files = [dict(file=p['name'].split('/')[-1], n=int(len(p['t'])), start=float(p['t'][0]), end=float(p['t'][-1]),
                  minute_stamps=p['minute_stamps'], hidden_gaps=p['hidden_gaps'], stray_dropped=p['stray_dropped'],
                  clock_set_back=p['clock_set_back'], replayed_dropped=p['replayed_dropped'], trimmed=p.get('trimmed', 0), dropped=p['n_dropped'], T_median=float(np.nanmedian(p['T'])) if np.isfinite(p['T']).any() else None)
             for p in parts]
    meta[name] = dict(samples=int(len(t)), files=len(parts), segments=int(seg[-1] + 1), breaks=nbrk,
                      hours=float((t[-1] - t[0]) / 3600), I_min=float(I.min()), I_max=float(I.max()), V_min=float(V.min()),
                      V_max=float(V.max()), T_min=float(np.nanmin(T)), T_max=float(np.nanmax(T)), file_table=files)


def rpcwby_histories(meta):
    W = pd.read_csv(WIN); K = pd.read_csv(KEYS).set_index('row_id')
    W = W[W.goal].copy(); W['last_row'] = [int(K.at[r, 'last_row']) for r in W.row_id]
    t_all, V_all, I_all, T_all, S_all, rec = [], [], [], [], [], []
    off = 0
    for (cell, member), g in W.groupby(['cell', 'member']):
        d = pd.read_csv(io.BytesIO(zipfile.ZipFile(RPC[cell]).read(member)), low_memory=False)
        d.columns = [c.replace('﻿', '').strip() for c in d.columns]
        # cell surface temperature = the first Aux_Temperature column (Test#1 _3, Test#2 _19, Test#3 thermocouple1); the second
        # column on Test#1/#2 (_38, _40) is the chamber, quantised at the set point (attempt03 took it; V84's labels are swapped)
        aux = [c for c in d.columns if c.startswith('Aux_Temperature')]
        tcol = aux[0] if aux else None
        if len(aux) > 1: assert d[aux[0]].nunique() > 10 * d[aux[1]].nunique(), (member, aux)
        tt = d['Test_Time(s)'].to_numpy(float); VV = d['Voltage(V)'].to_numpy(float); II = d['Current(A)'].to_numpy(float)
        TT = d[tcol].to_numpy(float) if tcol else np.full(len(d), np.nan)
        # wall clock minus Test_Time jumps where the test was paused or logging stopped (43 of 71 judged files): hidden gaps
        w = (pd.to_datetime(d['Date_Time'], format='mixed') - pd.Timestamp('1970-01-01')).dt.total_seconds().to_numpy()
        ta, hidden, _ = wall_time(tt, w, 30.)
        for r in g.itertuples():
            a, q, b = int(r.h0_row), int(r.query_first_row), int(r.last_row)
            sl = slice(a, b + 1); n = b - a + 1
            t_all.append(ta[sl]); V_all.append(VV[sl].astype(np.float32)); I_all.append(II[sl].astype(np.float32)); T_all.append(TT[sl].astype(np.float32))
            seg_r, _ = segments(ta[sl], II[sl], hidden[sl]); S_all.append(seg_r)
            rec.append(dict(row_id=r.row_id, cell=cell, member=member, direction=r.direction, soc_point=r.soc_point, offset=off, n=n,
                            query_index=q - a, h0_row=a, last_row=b, temp_col=tcol, T_median=float(np.nanmedian(TT[a:q])),
                            window_h=float((ta[q] - ta[a]) / 3600), hidden_gaps_in_history=int(hidden[a + 1:q + 1].sum()),
                            segments_in_history=int(seg_r[q - a] + 1)))
            off += n
    t = np.concatenate(t_all); np.savez(OUT / 'rpcwby_judged.npz', t=t, V=np.concatenate(V_all), I=np.concatenate(I_all), T=np.concatenate(T_all),
                                         segment=np.concatenate(S_all))
    R = pd.DataFrame(rec); R.to_csv(OUT / 'rpcwby_judged_index.csv', index=False)
    meta['rpcwby_judged'] = dict(rows=int(len(R)), samples=int(len(t)), temp_cols=sorted(R.temp_col.astype(str).unique().tolist()),
                                 window_h=dict(min=float(R.window_h.min()), max=float(R.window_h.max())))


def main():
    import sys
    if '--rpcwby-only' in sys.argv:  # rebuild only the evaluation histories (attempt03 used the chamber column)
        meta = json.load(open(OUT / 'summary.json')); rpcwby_histories(meta); json.dump(meta, open(OUT / 'summary.json', 'w'), indent=1)
        print(meta['rpcwby_judged']); return
    OUT.mkdir(exist_ok=False)
    A = pd.read_csv(AUDIT); S = A[A.kind == 'schedule'].copy()
    S['excluded'] = (S.bad_frac > 0) | (S.T_med < 15) | (S.T_med > 45)
    S.to_csv(OUT / 'uypydj_schedule_files.csv', index=False)
    zips = {z.name: z for z in (ROOT / 'raw/UYPYDJ').glob('*.zip')}
    jobs = []
    for r in S[~S.excluded].itertuples():
        zp = zips[r.zipname]; member = next(n for n in zipfile.ZipFile(zp).namelist() if n.endswith('/' + r.file))
        jobs.append((r.cell, str(zp), member))
    for d in sorted(p for p in MEND.iterdir() if p.is_dir()):
        for f in sorted(d.glob('*.mat')): jobs.append((f'mendeley_{d.name}', None, str(f)))
    with ProcessPoolExecutor(12) as ex: res = list(ex.map(read_mat, jobs, chunksize=2))
    meta = dict(skipped=[dict(src=r['src'], name=r['name'], why=r['skip']) for r in res if 'skip' in r],
                uypydj_excluded=S[S.excluded].file.tolist())
    by = {}
    for r in res:
        if 'skip' not in r: by.setdefault(r['src'], []).append(r)
    for name, parts in sorted(by.items()): pack(name, parts, meta)
    rpcwby_histories(meta)
    json.dump(meta, open(OUT / 'summary.json', 'w'), indent=1)
    for k, v in meta.items():
        if isinstance(v, dict): print(k, {kk: vv for kk, vv in v.items() if kk != 'file_table'})
    print('skipped', len(meta['skipped']), meta['skipped'][:5]); print('uypydj excluded', meta['uypydj_excluded'])


if __name__ == '__main__':
    main()
