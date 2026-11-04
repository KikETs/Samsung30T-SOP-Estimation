"""Rebuild the V273 centres from V272's saved local predictions and the fetched i3n tables, independently of strictfull_v273.py."""
from pathlib import Path
import numpy as np, pandas as pd
E = str(Path(__file__).resolve().parents[2]) + '/'
N5 = ['prev', 'L4', 'M6', 'DN6', 'G1']
cell = pd.read_csv(E + 'closedloop_v251/src/bench_rows.csv').set_index('row_id').evaluation_group
ph = {t: pd.read_csv(E + f'strictfull_v273/src_remote/{t}/pred.csv').query("status == 'ok'").set_index('row_id').pred_power_W for t in ('n_all', 'n_bA17', 'n_bA1')}
def lm(X):
    with np.errstate(all='ignore'):
        L = np.log(np.asarray(X, float)); L[~np.isfinite(L)] = np.nan; return np.nanmedian(L, axis=1)
C = pd.read_csv(E + 'leakfree_v272/centres_vpre_main.csv')
tab = lambda c, r: {'B': 'n_all', 'A1': 'n_bA17', 'A17': 'n_bA1'}[c] if cell[r] == 'B' else 'n_all'
C['i3n'] = [ph[tab(c, r)].get(r, np.nan) for c, r in zip(C.fold, C.row_id)]; C['m_chk'] = lm(C[N5 + ['i3n']])
F = pd.read_csv(E + 'strictfull_v273/centres_full.csv').merge(C[['fold', 'row_id', 'm_chk']], on=['fold', 'row_id'])
print('fold centres:', len(F), 'max |dm|', float(np.nanmax(np.abs(F.m - F.m_chk))), 'nan mismatch', int((F.m.isna() != F.m_chk.isna()).sum()))
S0 = pd.read_csv(E + 'leakfree_v272/strict_vpre_main.csv').pivot_table(index=['train_cell', 'row_id'], columns='name', values='pred_power_W').reset_index()
tab_s = lambda e, r: {'A17': 'n_bA17', 'A1': 'n_bA1'}[e] if cell[r] == 'B' else 'n_all'
S0['i3n'] = [ph[tab_s(e, r)].get(r, np.nan) for e, r in zip(S0.train_cell, S0.row_id)]; S0['m_chk'] = lm(S0[N5 + ['i3n']])
S = pd.read_csv(E + 'strictfull_v273/strict_full.csv').merge(S0[['train_cell', 'row_id', 'm_chk']], on=['train_cell', 'row_id'])
print('strict centres:', len(S), 'max |dm|', float(np.nanmax(np.abs(S.m - S.m_chk))), 'nan mismatch', int((S.m.isna() != S.m_chk.isna()).sum()))
# B rows: which table each context used, and how far the contexts differ
b = [r for r in cell.index if cell[r] == 'B' and r in ph['n_all'].index]
d17 = np.abs(np.log(ph['n_bA17'][b] / ph['n_all'][b])); d1 = np.abs(np.log(ph['n_bA1'][b] / ph['n_all'][b]))
print('B rows with i3n:', len(b), '| |log n_bA17/n_all| median', round(float(d17.median()), 4), 'max', round(float(d17.max()), 4), '| |log n_bA1/n_all| median', round(float(d1.median()), 4), 'max', round(float(d1.max()), 4))
print('rows per fold without any physics:', {f: int(g.i3n.isna().sum()) for f, g in C.groupby('fold')})
