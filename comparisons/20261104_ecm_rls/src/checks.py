"""Checks that do not touch the benchmark: synthetic identification (check 1) and solver unit tests.

    python -I -B src/checks.py          # prints and writes results/checks_synthetic.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm_rls as M                                      # noqa: E402
from evaluation import pilot_functions                   # noqa: E402

EXP = Path(__file__).resolve().parents[1]
SEED = 20261104


def step(t0, duration, current, gap_at=None):
    """Arbin-like logging of one step: samples at Step_Time 1, 2, ... duration s, an end duplicate at +0.001 s and a
    transitional sample at +0.004 s with the same current. gap_at: log only Step_Time 9.778 and the end."""
    st = np.arange(1.0, duration + 1e-9, 1.0)
    if gap_at is not None:
        st = np.array([gap_at, duration])
    st = np.r_[st, duration + 0.001, duration + 0.004]
    return t0 + st, np.full(len(st), float(current)), t0 + duration + 0.004


def synthetic_profile():
    """CC charge, rests, eight SOC steps (10-s pulse, 0.04-s zero-current transition, 3 A adjust, 300-s rest),
    one logging gap and one charge pulse; returns t, I and the index of the last sample."""
    T, C = [], []
    clock = 0.0
    for dur, cur, gap in [(1200, 1.5, None), (600, 0.0, None)]:
        t, i, clock = step(clock, dur, cur, gap)
        T.append(t)
        C.append(i)
    for k in range(8):
        pulse = 5.0 if k == 5 else -20.0
        for dur, cur, gap in [(10, pulse, 9.778 if k == 3 else None), (0.04, 0.0, None), (120, -3.0, None),
                              (300, 0.0, None)]:
            if dur < 1:
                t = clock + np.array([0.033, 0.043])
                i = np.zeros(2)
                clock += 0.043
            else:
                t, i, clock = step(clock, dur, cur, gap)
            T.append(t)
            C.append(i)
    return np.concatenate(T), np.concatenate(C)


def simulate_truth(t, I, E, r0, R, tau):
    x = np.zeros(len(tau))
    V = np.empty(len(t))
    for k in range(len(t)):
        if k:
            a = np.exp(-max(t[k] - t[k - 1], 0.0) / tau)
            x = a * x + (1 - a) * I[k]
        V[k] = E + r0 * I[k] + np.dot(R, x)
    return V, x


def closed_form(t, I, V, tau, lam, n_rc):
    """Weighted, prior-regularised least squares equal to RLS without the covariance bound."""
    x = np.zeros(len(tau))
    t_load = -np.inf
    Phi, Y, LK = [], [], []
    tp = None
    for k in range(len(t)):
        if tp is None:
            dt = 0.0
        else:
            dt = max(t[k] - tp, 0.0)
            a = np.exp(-dt / tau)
            x = a * x + (1 - a) * I[k]
        tp = t[k]
        if abs(I[k]) >= M.REST_A:
            t_load = t[k]
        if not (abs(I[k]) >= M.REST_A or t[k] - t_load <= M.RELAX_S):
            continue
        Phi.append(np.r_[1.0, I[k], x])
        Y.append(V[k])
        LK.append(lam ** dt)
    Phi, Y, LK = np.array(Phi), np.array(Y), np.array(LK)
    # weight of sample j at the end = product of lam_k for k > j; prior weight = product of all lam_k
    w = np.r_[np.cumprod(LK[::-1])[::-1][1:], 1.0]
    w0 = np.prod(LK)
    p0 = np.array([M.P0_E] + [M.P0_R] * (1 + n_rc))
    th0 = np.r_[V[0], np.zeros(1 + n_rc)]
    A = np.diag(w0 / p0) + (Phi * w[:, None]).T @ Phi
    b = (w0 / p0) * th0 + (Phi * w[:, None]).T @ Y
    return np.linalg.solve(A, b)


def identify(t, I, V, n_rc, lam, cap=True):
    arrays = {'syn': dict(t=t, I=I, V=V)}
    rd = M.Reader(arrays)
    snaps = M.run_pass(rd, 'syn', 0, [(len(t), 'end')], n_rc, (lam,), cap=cap)
    return snaps[len(t)][0]


def sop_from(E, r0, R, tau, z, discharge, h=10.0, steps=801):
    return float(M.rc_power_n(np.array([E]), np.array([r0]), np.array([R]), np.array([tau]), np.array([z]),
                              np.array([discharge]), np.array([h]), steps)[0])


def independent_power_n(E, r0, R, tau, z0, discharge, h):
    """solve_ivp + bisection reference for the n-branch constant-power problem (run_pilot.independent_power
    generalised)."""
    sign = -1 if discharge else 1

    def trial(power):
        def calc(z):
            u = E + np.sum(z, axis=0)
            disc = u ** 2 + 4 * sign * r0 * power
            v = .5 * (u + np.sqrt(np.maximum(disc, 0)))
            return v, power / np.maximum(v, 1e-12), disc

        def rhs(_, z):
            _, current, _ = calc(z)
            return (sign * R * current - z) / tau
        sol = solve_ivp(rhs, (0, h), z0, rtol=1e-9, atol=1e-11, dense_output=True, method='LSODA')
        if not sol.success:
            return False
        v, i, disc = calc(sol.sol(np.linspace(0, h, 1001)))
        return bool(np.all(disc >= 0) and np.all((v >= 2.55) & (i <= 30) if discharge else (v <= 4.15) & (i <= 15)))
    lo = 0.
    hi = (30 * max(E + np.sum(z0), 0) if discharge else 15 * 4.15) * 1.001
    if not trial(0):
        return 0.
    for _ in range(24):
        mid = (lo + hi) / 2
        if trial(mid):
            lo = mid
        else:
            hi = mid
    return lo


def synthetic_checks():
    t, I = synthetic_profile()
    E = 3.7
    truths = {1: dict(r0=0.020, R=np.array([0.015]), j=np.array([10])),
              2: dict(r0=0.020, R=np.array([0.010, 0.025]), j=np.array([7, 15]))}
    rng = np.random.default_rng(SEED)
    res = {}
    for n_rc, tr in truths.items():
        tau = M.TAU_GRID[tr['j']]
        V, x_end = simulate_truth(t, I, E, tr['r0'], tr['R'], tau)
        sop_true_dis = sop_from(E, tr['r0'], tr['R'], tau, tr['R'] * x_end, True)
        sop_true_chg = sop_from(E, tr['r0'], tr['R'], tau, tr['R'] * x_end, False)
        out = {}
        for noise in (0.0, 0.001):
            Vn = V + (rng.normal(0, noise, len(V)) if noise else 0.0)
            snap = identify(t, I, Vn, n_rc, 0.999)
            jsel = np.searchsorted(M.TAU_GRID, snap['tau'] - 1e-12)
            th = snap['theta']
            Rhat = th[2:]
            sop_dis = sop_from(snap['E_anchor'], th[1], Rhat, snap['tau'], snap['z0'], True)
            sop_chg = sop_from(snap['E_anchor'], th[1], Rhat, snap['tau'], snap['z0'], False)
            o = dict(status=snap['status'], tau_index=jsel.tolist(), true_tau_index=tr['j'].tolist(),
                     R0=float(th[1]), R=Rhat.tolist(), R0_rel_err=float(abs(th[1] / tr['r0'] - 1)),
                     R_rel_err=np.abs(Rhat / tr['R'] - 1).tolist(), E_anchor_err_V=float(snap['E_anchor'] - E),
                     sop_dis_W=sop_dis, sop_dis_true_W=sop_true_dis, sop_chg_W=sop_chg, sop_chg_true_W=sop_true_chg,
                     n_updates=snap['n_updates'], n_cap=snap['n_cap'])
            if noise == 0.0:
                o['passed'] = bool(snap['status'] == 'ok' and np.array_equal(jsel, tr['j'])
                                   and o['R0_rel_err'] <= 1e-3 and max(o['R_rel_err']) <= 1e-3
                                   and abs(o['E_anchor_err_V']) <= 1e-4
                                   and abs(sop_dis - sop_true_dis) <= 0.05 and abs(sop_chg - sop_true_chg) <= 0.05)
            else:
                o['passed'] = bool(snap['status'] == 'ok' and np.all(np.abs(jsel - tr['j']) <= 1)
                                   and o['R0_rel_err'] <= 0.02 and max(o['R_rel_err']) <= 0.15
                                   and abs(sop_dis / sop_true_dis - 1) <= 0.02 and abs(sop_chg / sop_true_chg - 1) <= 0.02)
            out[f'noise_{noise * 1000:g}mV'] = o
        # RLS without the covariance bound equals the closed-form weighted regularised least squares
        idx = M.member_taus(n_rc)
        m_true = int(np.flatnonzero((idx == tr['j']).all(1))[0])
        arrays = {'syn': dict(t=t, I=I, V=V + rng.normal(0, 0.001, len(V)))}
        lam = 0.998
        snap_nc = M.run_pass(M.Reader(arrays), 'syn', 0, [(len(t), 'end')], n_rc, M.LAMBDAS, cap=False,
                             return_bank=True)[len(t)][M.LAMBDAS.index(lam)]
        Vn = arrays['syn']['V']
        th_cf = closed_form(t, I, Vn, tau, lam, n_rc)
        th_bank = snap_nc['theta_bank'][m_true]
        th_ind = rls_member(t, I, Vn, tau, lam, n_rc)

        def rel(a, b):
            return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        out['rls_equals_closed_form'] = dict(member=m_true, lam=lam, noise_mV=1.0,
                                             bank_vs_closed_form_max_rel_diff=rel(th_bank, th_cf),
                                             plain_rls_vs_closed_form_max_rel_diff=rel(th_ind, th_cf))
        out['rls_equals_closed_form']['passed'] = bool(max(rel(th_bank, th_cf), rel(th_ind, th_cf)) <= 1e-8)
        single = M.run_pass(M.Reader(arrays), 'syn', 0, [(len(t), 'end')], n_rc, (lam,), cap=True,
                            return_bank=True)[len(t)][0]
        stacked = M.run_pass(M.Reader(arrays), 'syn', 0, [(len(t), 'end')], n_rc, M.LAMBDAS, cap=True,
                             return_bank=True)[len(t)][M.LAMBDAS.index(lam)]
        d = float(np.max(np.abs(single['theta_bank'] - stacked['theta_bank'])))
        out['stacked_lambdas_equal_single'] = dict(max_abs_theta_diff=d, passed=bool(d <= 1e-12))
        res[f'{n_rc}RC'] = out
    return res


def rls_member(t, I, V, tau, lam, n_rc):
    """Plain RLS (no bound) for one fixed member, written independently of run_pass."""
    n = 2 + n_rc
    p0 = np.array([M.P0_E] + [M.P0_R] * (1 + n_rc))
    P = np.diag(p0)
    th = np.r_[V[0], np.zeros(1 + n_rc)]
    x = np.zeros(len(tau))
    t_load = -np.inf
    tp = None
    for k in range(len(t)):
        if tp is None:
            dt = 0.0
        else:
            dt = max(t[k] - tp, 0.0)
            a = np.exp(-dt / tau)
            x = a * x + (1 - a) * I[k]
        tp = t[k]
        if abs(I[k]) >= M.REST_A:
            t_load = t[k]
        if not (abs(I[k]) >= M.REST_A or t[k] - t_load <= M.RELAX_S):
            continue
        lk = lam ** dt
        phi = np.r_[1.0, I[k], x]
        g = P @ phi / (lk + phi @ P @ phi)
        th = th + g * (V[k] - phi @ th)
        P = (P - np.outer(g, phi @ P)) / lk
    return th


def solver_checks():
    F, _ = pilot_functions()
    rng = np.random.default_rng(SEED)
    n = 2000
    r0 = rng.uniform(0.005, 0.08, n)
    r1 = rng.uniform(0.001, 0.08, n)
    tau = 10 ** rng.uniform(-1, 3, n)
    E = rng.uniform(3.0, 4.15, n)
    z = rng.uniform(-0.08, 0.08, n)
    dis = rng.random(n) < 0.5
    h = np.where(rng.random(n) < 0.5, 10.0, 2.0)
    pilot = F['rc_power'](np.stack([r0, r1, tau, z, E], 1), dis, h, 201)
    mine = M.rc_power_n(E, r0, r1[:, None], tau[:, None], z[:, None], dis, h, 201)
    eq = float(np.nanmax(np.abs(pilot - mine)))
    # resistance-only limit against the closed form I* = min(headroom / R0, I_max), P = I* (E -/+ R0 I*)
    tiny = np.full((n, 1), 1e-12)
    rp = M.rc_power_n(E, r0, tiny, np.ones((n, 1)), np.zeros((n, 1)), dis, h, 201)
    head = np.where(dis, E - 2.55, 4.15 - E)
    cur = np.minimum(np.maximum(head, 0) / r0, np.where(dis, 30., 15.))
    closed = cur * (E + np.where(dis, -1, 1) * cur * r0)
    an = float(np.max(np.abs(rp - closed) - (1e-5 * closed + 1e-5)))
    # independent ODE reference for 2RC parameter sets
    m = 20
    rr = np.stack([rng.uniform(0.002, 0.05, m), rng.uniform(0.002, 0.05, m)], 1)
    tt = np.sort(10 ** rng.uniform(-1, 3, (m, 2)), 1)
    zz = rng.uniform(-0.05, 0.05, (m, 2))
    EE = rng.uniform(3.2, 4.1, m)
    r00 = rng.uniform(0.01, 0.05, m)
    dd = np.arange(m) % 2 == 0
    hh = np.full(m, 10.0)
    p801 = M.rc_power_n(EE, r00, rr, tt, zz, dd, hh, 801)
    ind = np.array([independent_power_n(EE[i], r00[i], rr[i], tt[i], zz[i], dd[i], hh[i]) for i in range(m)])
    return dict(n1_equals_pilot_rc_power=dict(cases=n, max_abs_diff_W=eq, passed=bool(eq <= 1e-12)),
                resistance_limit_closed_form=dict(cases=n, max_excess_over_tolerance_W=an, passed=bool(an <= 0)),
                independent_ode_2rc=dict(cases=m, max_abs_diff_W=float(np.max(np.abs(p801 - ind))),
                                         passed=bool(np.max(np.abs(p801 - ind)) <= 0.05)))


def main():
    res = dict(synthetic_identification=synthetic_checks(), solver=solver_checks())
    passed = all(v['passed'] for k, d in res['synthetic_identification'].items() for v in d.values()) and \
        all(v['passed'] for v in res['solver'].values())
    res['all_passed'] = bool(passed)
    (EXP / 'results').mkdir(exist_ok=True)
    (EXP / 'results/checks_synthetic.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    main()
