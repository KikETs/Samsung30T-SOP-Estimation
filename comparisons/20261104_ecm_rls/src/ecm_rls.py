"""Classical Thevenin equivalent-circuit SOP predictor (1RC / 2RC) with online RLS identification.

Model (raw current sign: + charge, - discharge; volts, amperes, seconds, ohms):
    V_k = E + R0 I_k + sum_i R_i x_{i,k}
    x_{i,k} = a_{i,k} x_{i,k-1} + (1 - a_{i,k}) I_k,   a_{i,k} = exp(-dt_k / tau_i),  dt_k = max(t_k - t_{k-1}, 0)
The RC states use an exact zero-order hold in which the current of sample k is held over (t_{k-1}, t_k]
(right hold): the Arbin log records the first sample of a step about 1 s after the step started (Step_Time ~ 1 s),
so the current of a sample is the current that flowed during the interval that ends at it.

Identification: for every member of a fixed time-constant bank (1RC: 21 log-spaced tau in 0.1..1000 s; 2RC: pairs
tau1 < tau2 at least three grid steps apart), the linear parameters theta = [E, R0, R1(, R2)] are identified by
exponentially weighted RLS with forgetting factor lam per second (lam_k = lam ** dt_k). E is a slowly varying bias
(open-circuit voltage) tracked through the forgetting. The member with the smallest exponentially weighted sum of
squared a-priori errors among members with valid parameters is selected at each snapshot.

The predictor reads raw rows only through Reader.read(member, lo, hi), which logs every call. One causal pass per
sweep (rows sharing member and h0_row) reads [h0, s1), [s1, s2), ... and takes a snapshot just before each requested
index s (state after samples < s).
"""
import numpy as np

TAU_GRID = 10.0 ** (-1.0 + 0.2 * np.arange(21))          # 0.1 ... 1000 s
MIN_PAIR_STEPS = 3                                      # 2RC: tau2 / tau1 >= 10 ** 0.6
REST_A = 0.02                                           # |I| below this is rest
RELAX_S = 300.0                                         # rest samples within this time after a load are used
EXC_A = 0.5                                             # excitation: at least one gated sample with |I| >= 0.5 A
MIN_UPDATES = 20                                        # ... and at least 20 gated updates since h0
R_MAX = 1.0                                             # valid: 0 < R0, R_i <= 1 ohm
P0_E, P0_R = 1.0, 1.0                                   # initial covariance diag (V^2, ohm^2)
LAMBDAS = (0.99, 0.995, 0.998, 0.999, 0.9995)           # forgetting factor per second (hyperparameter grid)
V_DIS, V_CHG, I_DIS, I_CHG = 2.55, 4.15, 30.0, 15.0


def member_taus(n_rc):
    """Indices into TAU_GRID for every bank member, shape (M, n_rc)."""
    if n_rc == 1:
        return np.arange(len(TAU_GRID))[:, None]
    return np.array([(a, b) for a in range(len(TAU_GRID)) for b in range(a + MIN_PAIR_STEPS, len(TAU_GRID))])


class Reader:
    """Raw Test_Time, Current, Voltage of member files. Every read of rows [lo, hi) is logged."""

    def __init__(self, arrays, log=None):
        self.arrays = arrays                            # member -> {'t','I','V'} (possibly truncated copies)
        self.log = [] if log is None else log

    def read(self, member, lo, hi, purpose):
        d = self.arrays[member]
        n = len(d['t'])
        if not (0 <= lo <= hi <= n):
            raise IndexError(f'read [{lo}, {hi}) outside available rows [0, {n}) of {member}')
        self.log.append(('read', member, int(lo), int(hi), purpose))
        return d['t'][lo:hi].copy(), d['I'][lo:hi].copy(), d['V'][lo:hi].copy()


def run_pass(reader, member, h0, snaps, n_rc, lams, cap=True, purpose='sweep', return_bank=False):
    """One causal RLS pass from h0 for every forgetting factor in lams at once (independent banks that share the
    data and the RC regressors). snaps: list of (s, label) with increasing s > h0; the snapshot at s uses rows
    [h0, s) only. Returns {s: [snapshot for each lam]}. With cap=True, for a bank member whose covariance diagonal
    would exceed its initial value after the division by lam_k, the division is skipped at that sample."""
    lams = np.atleast_1d(np.asarray(lams, dtype=float))
    L = len(lams)
    idx = member_taus(n_rc)
    M, n = len(idx), 2 + n_rc
    LM = L * M
    block = np.repeat(np.arange(L), M)
    idx_all = np.tile(idx, (L, 1))
    p0 = np.array([P0_E] + [P0_R] * (1 + n_rc))
    theta = np.zeros((LM, n))
    P = np.broadcast_to(np.diag(p0), (LM, n, n)).copy()
    J = np.zeros(LM)
    phi = np.ones((LM, n))
    x = np.zeros(len(TAU_GRID))
    t_prev = None
    t_load = -np.inf
    n_upd = n_exc = 0
    n_cap = np.zeros(L, dtype=int)
    lv_has = np.zeros(L, bool)
    lv_m = np.zeros(L, dtype=int)
    lv_theta = np.zeros((L, n))
    lv_J = np.zeros(L)
    out = {}
    cursor = h0
    for s, label in snaps:
        if s <= cursor:
            raise ValueError('snapshot indices must increase and exceed h0')
        t, I, V = reader.read(member, cursor, s, purpose)
        for k in range(len(t)):
            tk, Ik, Vk = t[k], I[k], V[k]
            if t_prev is None:
                theta[:, 0] = Vk                         # prior E = first voltage of the window
                dt = 0.0
            else:
                dt = tk - t_prev
                if dt < 0.0:
                    dt = 0.0
                a = np.exp(-dt / TAU_GRID)
                x = a * x + (1.0 - a) * Ik
            t_prev = tk
            if abs(Ik) >= REST_A:
                t_load = tk
            if not (abs(Ik) >= REST_A or tk - t_load <= RELAX_S):
                continue                                 # rest without excitation: no update, no forgetting
            lk = (lams ** dt)[block]
            phi[:, 1] = Ik
            phi[:, 2:] = x[idx_all]
            Pphi = np.matmul(P, phi[:, :, None])[:, :, 0]
            den = lk + np.einsum('mi,mi->m', phi, Pphi)
            e = Vk - np.einsum('mi,mi->m', phi, theta)
            good = np.isfinite(den) & (den > 0) & np.isfinite(e)
            if not good.all():
                Pphi = np.where(good[:, None], Pphi, 0.0)
                den = np.where(good, den, 1.0)
                e = np.where(good, e, 0.0)
            K = Pphi / den[:, None]
            theta += K * e[:, None]
            P -= K[:, :, None] * Pphi[:, None, :]
            if cap:
                over = (np.einsum('mii->mi', P) > p0[None, :] * lk[:, None]).any(1)
                n_cap += over.reshape(L, M).any(1)
                P /= np.where(over, 1.0, lk)[:, None, None]
            else:
                P /= lk[:, None, None]
            J = lk * J + e * e
            n_upd += 1
            if abs(Ik) >= EXC_A:
                n_exc += 1
            valid = np.isfinite(theta).all(1) & (theta[:, 1:] > 0).all(1) & (theta[:, 1:] <= R_MAX).all(1)
            vb = valid.reshape(L, M)
            anyv = vb.any(1)
            if anyv.any():
                mb = np.where(vb, J.reshape(L, M), np.inf).argmin(1)
                rows = np.arange(L) * M + mb
                lv_theta = np.where(anyv[:, None], theta[rows], lv_theta)
                lv_m = np.where(anyv, mb, lv_m)
                lv_J = np.where(anyv, J[rows], lv_J)
                lv_has |= anyv
        cursor = s
        reader.log.append(('snap', member, int(s), label))
        valid = np.isfinite(theta).all(1) & (theta[:, 1:] > 0).all(1) & (theta[:, 1:] <= R_MAX).all(1)
        snaps_l = []
        for l in range(L):
            b = slice(l * M, (l + 1) * M)
            vb = valid[b]
            snap = dict(s=int(s), label=label, lam=float(lams[l]), n_updates=n_upd, n_excited=n_exc,
                        n_cap=int(n_cap[l]), t_last=float(t_prev), V_last=float(V[-1]), I_last=float(I[-1]),
                        n_valid_members=int(vb.sum()))
            if return_bank:
                snap['theta_bank'] = theta[b].copy()
                snap['J_bank'] = J[b].copy()
            if n_upd < MIN_UPDATES or n_exc < 1:
                snap.update(status='missing', reason='insufficient excitation')
            elif vb.any():
                m = int(np.argmin(np.where(vb, J[b], np.inf)))
                snap.update(status='ok', member=m, theta=theta[b][m].copy(), J=float(J[b][m]))
            elif lv_has[l]:
                snap.update(status='fallback', member=int(lv_m[l]), theta=lv_theta[l].copy(), J=float(lv_J[l]))
            else:
                snap.update(status='missing', reason='no valid estimate')
            if snap['status'] != 'missing':
                ti = idx[snap['member']]
                snap['tau'] = TAU_GRID[ti].copy()
                snap['x'] = x[ti].copy()
                th = snap['theta']
                snap['E_anchor'] = float(snap['V_last'] - th[1] * snap['I_last'] - np.dot(th[2:], snap['x']))
                snap['z0'] = th[2:] * snap['x']
            snaps_l.append(snap)
        out[s] = snaps_l
    return out


def simulate_feasible_n(P, E, r0, R, TAU, Z, discharge, h, steps):
    """run_pilot.simulate_feasible generalised to n RC branches: R, TAU, Z have shape (rows, n). For n = 1 the
    arithmetic is the same operation sequence as the pilot function."""
    sign = np.where(discharge, -1., 1.)
    valid = np.ones(len(P), bool)
    dt = h / (steps - 1)
    decay = np.exp(-dt[:, None] / TAU)
    half = np.exp(-.5 * dt[:, None] / TAU)
    z = Z.copy()

    def electrical(pol):
        u = E + pol
        disc = u * u + 4 * sign * r0 * P
        v = .5 * (u + np.sqrt(np.maximum(disc, 0)))
        current = P / np.maximum(v, 1e-12)
        good = (disc >= 0) & (v > 0) & np.where(discharge, (v >= V_DIS) & (current <= I_DIS),
                                                  (v <= V_CHG) & (current <= I_CHG))
        return current, good
    for j in range(steps):
        current, good = electrical(z.sum(1))
        valid &= good
        if j == steps - 1:
            break
        mid = z * half + sign[:, None] * R * current[:, None] * (1 - half)
        current_mid, good_mid = electrical(mid.sum(1))
        valid &= good_mid
        z = z * decay + sign[:, None] * R * current_mid[:, None] * (1 - decay)
    return valid


def rc_power_n(E, r0, R, TAU, Z, discharge, h, steps=201):
    """run_pilot.rc_power generalised to n RC branches: bisection (24 halvings) on constant power."""
    out = np.full(len(h), np.nan)
    mask = np.isfinite(E) & np.isfinite(r0) & np.isfinite(R).all(1) & np.isfinite(TAU).all(1) & np.isfinite(Z).all(1)
    EE, rr, RR, TT, ZZ, dd, hh = E[mask], r0[mask], R[mask], TAU[mask], Z[mask], discharge[mask], h[mask]
    low = np.zeros(len(EE))
    high = np.where(dd, 30 * np.maximum(EE + ZZ.sum(1), 0), 15 * 4.15) * 1.001
    zero = simulate_feasible_n(low, EE, rr, RR, TT, ZZ, dd, hh, steps)
    for _ in range(24):
        mid = (low + high) / 2
        good = simulate_feasible_n(mid, EE, rr, RR, TT, ZZ, dd, hh, steps)
        low = np.where(good, mid, low)
        high = np.where(good, high, mid)
    out[mask] = np.where(zero, low, 0)
    return out


def snapshot_arrays(snaps, n_rc):
    """Stack snapshots into solver arrays; missing snapshots give NaN rows (no prediction)."""
    k = len(snaps)
    E = np.full(k, np.nan)
    r0 = np.full(k, np.nan)
    R = np.full((k, n_rc), np.nan)
    TAU = np.full((k, n_rc), np.nan)
    Z = np.full((k, n_rc), np.nan)
    for i, s in enumerate(snaps):
        if s['status'] == 'missing':
            continue
        E[i] = s['E_anchor']
        r0[i] = s['theta'][1]
        R[i] = s['theta'][2:]
        TAU[i] = s['tau']
        Z[i] = s['z0']
    return E, r0, R, TAU, Z


def simulate_measured(snap, t, I):
    """Voltage of a measured current sequence (samples t, I following the snapshot index) from a snapshot."""
    th = snap['theta']
    tau = snap['tau']
    x = snap['x'].copy()
    E = snap['E_anchor']
    tp = snap['t_last']
    out = np.empty(len(t))
    for k in range(len(t)):
        dt = max(t[k] - tp, 0.0)
        a = np.exp(-dt / tau)
        x = a * x + (1.0 - a) * I[k]
        out[k] = E + th[1] * I[k] + np.dot(th[2:], x)
        tp = t[k]
    return out
