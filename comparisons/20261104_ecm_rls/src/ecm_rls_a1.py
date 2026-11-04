"""Amendment 1 (PROTOCOL_AMENDMENTS.md): candidates ECM-1RC-RLS-A1 and ECM-2RC-RLS-A1.

Differences from the pre-registered ecm_rls.py (which is left unchanged):
  * stabilised exponential forgetting in information form (Kulhavy and Zarrop 1993, forgetting towards the prior
    information) instead of the covariance bound:
        R_k = lam_k R_{k-1} + (1 - lam_k) R_prior + phi phi^T,   b_k = lam_k b_{k-1} + (1 - lam_k) R_prior theta_0 + phi V_k,
        theta_k = R_k^{-1} b_k;
  * an optional local open-circuit-voltage slope kappa on the charge throughput Q (Ah since h0 from the measured
    current, right hold), chosen together with lam per evaluation cell by the voltage rule:
        V_k = E + kappa Q_k + R0 I_k + sum_i R_i x_{i,k};
  * SOP and voltage simulation with the open-circuit voltage E_q + kappa q(t) during the pulse.
Bank, gating, excitation, validity of R, selection by the a-priori error, fallback and anchoring are those of
ecm_rls.py.
"""
import numpy as np

from ecm_rls import (EXC_A, I_CHG, I_DIS, LAMBDAS, MIN_UPDATES, P0_E, P0_R, R_MAX, RELAX_S, REST_A, TAU_GRID, V_CHG,
                     V_DIS, member_taus)

KAPPA_MAX = 20.0                                          # valid: 0 <= kappa <= 20 V/Ah
P0_KAPPA = 25.0                                           # prior variance of kappa, (V/Ah)^2
CONFIGS = tuple((lam, kap) for kap in (False, True) for lam in LAMBDAS)   # snapshot order of run_pass


class _Bank:
    """All members of one structure (kappa off or on) for every forgetting factor."""

    def __init__(self, idx, lams, use_kappa):
        self.idx, self.lams, self.kap = idx, np.asarray(lams, dtype=float), use_kappa
        self.L, self.M, self.n_rc = len(self.lams), len(idx), idx.shape[1]
        self.rpos = 2 if use_kappa else 1                 # theta = [E, (kappa,) R0, R1(, R2)]
        self.n = self.rpos + 1 + self.n_rc
        LM = self.L * self.M
        self.block = np.repeat(np.arange(self.L), self.M)
        self.idx_all = np.tile(idx, (self.L, 1))
        p0 = [P0_E] + ([P0_KAPPA] if use_kappa else []) + [P0_R] * (1 + self.n_rc)
        self.R_prior = np.diag(1.0 / np.array(p0))
        self.Rinfo = np.broadcast_to(self.R_prior, (LM, self.n, self.n)).copy()
        self.binfo = np.zeros((LM, self.n))
        self.theta = np.zeros((LM, self.n))
        self.b_prior = np.zeros(self.n)
        self.J = np.zeros(LM)
        self.phi = np.ones((LM, self.n))
        self.lv_has = np.zeros(self.L, bool)
        self.lv_m = np.zeros(self.L, dtype=int)
        self.lv_theta = np.zeros((self.L, self.n))
        self.lv_J = np.zeros(self.L)

    def init(self, V0):
        th0 = np.zeros(self.n)
        th0[0] = V0
        self.b_prior = self.R_prior @ th0
        self.theta[:] = th0
        self.binfo[:] = self.b_prior

    def valid(self):
        r = self.theta[:, self.rpos:]
        v = np.isfinite(self.theta).all(1) & (r > 0).all(1) & (r <= R_MAX).all(1)
        if self.kap:
            v &= (self.theta[:, 1] >= 0) & (self.theta[:, 1] <= KAPPA_MAX)
        return v

    def update(self, dt, Ik, Vk, x, Qk):
        lk = (self.lams ** dt)[self.block]
        phi = self.phi
        if self.kap:
            phi[:, 1] = Qk
        phi[:, self.rpos] = Ik
        phi[:, self.rpos + 1:] = x[self.idx_all]
        e = Vk - np.einsum('mi,mi->m', phi, self.theta)   # a-priori error
        self.J = lk * self.J + e * e
        self.Rinfo = (lk[:, None, None] * self.Rinfo + (1.0 - lk)[:, None, None] * self.R_prior
                      + phi[:, :, None] * phi[:, None, :])
        self.binfo = lk[:, None] * self.binfo + (1.0 - lk)[:, None] * self.b_prior + phi * Vk
        self.theta = np.linalg.solve(self.Rinfo, self.binfo[:, :, None])[:, :, 0]
        vb = self.valid().reshape(self.L, self.M)
        anyv = vb.any(1)
        if anyv.any():
            mb = np.where(vb, self.J.reshape(self.L, self.M), np.inf).argmin(1)
            rows = np.arange(self.L) * self.M + mb
            self.lv_theta = np.where(anyv[:, None], self.theta[rows], self.lv_theta)
            self.lv_m = np.where(anyv, mb, self.lv_m)
            self.lv_J = np.where(anyv, self.J[rows], self.lv_J)
            self.lv_has |= anyv

    def snapshot(self, l, x, base, return_bank):
        b = slice(l * self.M, (l + 1) * self.M)
        v = self.valid()[b]
        snap = dict(base, lam=float(self.lams[l]), kappa_on=bool(self.kap), n_valid_members=int(v.sum()), n_cap=0)
        if return_bank:
            snap['theta_bank'] = self.theta[b].copy()
            snap['J_bank'] = self.J[b].copy()
        if base['n_updates'] < MIN_UPDATES or base['n_excited'] < 1:
            snap.update(status='missing', reason='insufficient excitation')
            return snap
        if v.any():
            m = int(np.argmin(np.where(v, self.J[b], np.inf)))
            snap.update(status='ok', member=m, theta=self.theta[b][m].copy(), J=float(self.J[b][m]))
        elif self.lv_has[l]:
            snap.update(status='fallback', member=int(self.lv_m[l]), theta=self.lv_theta[l].copy(),
                        J=float(self.lv_J[l]))
        else:
            snap.update(status='missing', reason='no valid estimate')
            return snap
        th = snap['theta']
        ti = self.idx[snap['member']]
        snap['tau'] = TAU_GRID[ti].copy()
        snap['x'] = x[ti].copy()
        snap['E_rls'] = float(th[0])
        snap['kappa'] = float(th[1]) if self.kap else 0.0
        snap['R0'] = float(th[self.rpos])
        snap['R'] = th[self.rpos + 1:].copy()
        snap['E_anchor'] = float(snap['V_last'] - snap['R0'] * snap['I_last'] - np.dot(snap['R'], snap['x']))
        snap['z0'] = snap['R'] * snap['x']
        return snap


def run_pass(reader, member, h0, snaps, n_rc, lams=LAMBDAS, purpose='sweep', return_bank=False):
    """One causal pass from h0 with both structures for every lam. Returns {s: [snapshot for each (lam, kappa) in
    CONFIGS order]}; the snapshot at s uses rows [h0, s) only."""
    idx = member_taus(n_rc)
    banks = [_Bank(idx, lams, False), _Bank(idx, lams, True)]
    x = np.zeros(len(TAU_GRID))
    Q = 0.0
    t_prev = None
    t_load = -np.inf
    n_upd = n_exc = 0
    out = {}
    cursor = h0
    for s, label in snaps:
        if s <= cursor:
            raise ValueError('snapshot indices must increase and exceed h0')
        t, I, V = reader.read(member, cursor, s, purpose)
        for k in range(len(t)):
            tk, Ik, Vk = t[k], I[k], V[k]
            if t_prev is None:
                for bk in banks:
                    bk.init(Vk)
                dt = 0.0
            else:
                dt = tk - t_prev
                if dt < 0.0:
                    dt = 0.0
                a = np.exp(-dt / TAU_GRID)
                x = a * x + (1.0 - a) * Ik
                Q += Ik * dt / 3600.0
            t_prev = tk
            if abs(Ik) >= REST_A:
                t_load = tk
            if not (abs(Ik) >= REST_A or tk - t_load <= RELAX_S):
                continue
            for bk in banks:
                bk.update(dt, Ik, Vk, x, Q)
            n_upd += 1
            if abs(Ik) >= EXC_A:
                n_exc += 1
        cursor = s
        reader.log.append(('snap', member, int(s), label))
        base = dict(s=int(s), label=label, n_updates=n_upd, n_excited=n_exc, t_last=float(t_prev),
                    V_last=float(V[-1]), I_last=float(I[-1]), Q_last=float(Q))
        out[s] = [bk.snapshot(l, x, base, return_bank) for bk in banks for l in range(bk.L)]
    return out


def simulate_feasible_nk(P, E, r0, R, TAU, Z, K, discharge, h, steps):
    """ecm_rls.simulate_feasible_n with the open-circuit voltage E + K q(t), q = charge moved since the pulse start
    (Ah, raw sign). Same constraints and midpoint scheme; for K = 0 the same values as simulate_feasible_n."""
    sign = np.where(discharge, -1., 1.)
    valid = np.ones(len(P), bool)
    dt = h / (steps - 1)
    decay = np.exp(-dt[:, None] / TAU)
    half = np.exp(-.5 * dt[:, None] / TAU)
    z = Z.copy()
    q = np.zeros(len(P))

    def electrical(pol, qq):
        u = E + K * qq + pol
        disc = u * u + 4 * sign * r0 * P
        v = .5 * (u + np.sqrt(np.maximum(disc, 0)))
        current = P / np.maximum(v, 1e-12)
        good = (disc >= 0) & (v > 0) & np.where(discharge, (v >= V_DIS) & (current <= I_DIS),
                                                  (v <= V_CHG) & (current <= I_CHG))
        return current, good
    for j in range(steps):
        current, good = electrical(z.sum(1), q)
        valid &= good
        if j == steps - 1:
            break
        mid = z * half + sign[:, None] * R * current[:, None] * (1 - half)
        q_mid = q + sign * current * (.5 * dt) / 3600.0
        current_mid, good_mid = electrical(mid.sum(1), q_mid)
        valid &= good_mid
        z = z * decay + sign[:, None] * R * current_mid[:, None] * (1 - decay)
        q = q + sign * current_mid * dt / 3600.0
    return valid


def rc_power_nk(E, r0, R, TAU, Z, K, discharge, h, steps=201):
    """ecm_rls.rc_power_n with the open-circuit-voltage slope K (V/Ah); same bisection and bounds."""
    out = np.full(len(h), np.nan)
    mask = (np.isfinite(E) & np.isfinite(r0) & np.isfinite(R).all(1) & np.isfinite(TAU).all(1)
            & np.isfinite(Z).all(1) & np.isfinite(K))
    EE, rr, RR, TT, ZZ, KK, dd, hh = E[mask], r0[mask], R[mask], TAU[mask], Z[mask], K[mask], discharge[mask], h[mask]
    low = np.zeros(len(EE))
    high = np.where(dd, 30 * np.maximum(EE + ZZ.sum(1), 0), 15 * 4.15) * 1.001
    zero = simulate_feasible_nk(low, EE, rr, RR, TT, ZZ, KK, dd, hh, steps)
    for _ in range(24):
        mid = (low + high) / 2
        good = simulate_feasible_nk(mid, EE, rr, RR, TT, ZZ, KK, dd, hh, steps)
        low = np.where(good, mid, low)
        high = np.where(good, high, mid)
    out[mask] = np.where(zero, low, 0)
    return out


def snapshot_arrays(snaps, n_rc):
    k = len(snaps)
    E, r0, K = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    R, TAU, Z = np.full((k, n_rc), np.nan), np.full((k, n_rc), np.nan), np.full((k, n_rc), np.nan)
    for i, s in enumerate(snaps):
        if s['status'] == 'missing':
            continue
        E[i], r0[i], K[i] = s['E_anchor'], s['R0'], s['kappa']
        R[i], TAU[i], Z[i] = s['R'], s['tau'], s['z0']
    return E, r0, R, TAU, Z, K


def simulate_measured(snap, t, I):
    """Voltage of a measured current sequence following the snapshot index."""
    tau, R = snap['tau'], snap['R']
    x = snap['x'].copy()
    E, kap, r0 = snap['E_anchor'], snap['kappa'], snap['R0']
    tp = snap['t_last']
    q = 0.0
    out = np.empty(len(t))
    for k in range(len(t)):
        dt = max(t[k] - tp, 0.0)
        a = np.exp(-dt / tau)
        x = a * x + (1.0 - a) * I[k]
        q += I[k] * dt / 3600.0
        out[k] = E + kap * q + r0 * I[k] + np.dot(R, x)
        tp = t[k]
    return out
