"""Synthetic and solver checks of amendment 1 (no benchmark data).

    python -I -B src/checks_a1.py     # writes results/checks_synthetic_a1.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm_rls as M                                       # noqa: E402
import ecm_rls_a1 as A                                    # noqa: E402
from checks import step, synthetic_profile                # noqa: E402

EXP = Path(__file__).resolve().parents[1]
SEED = 20261104


def drift_profile():
    """3000-s constant-current charge (wind-up stress), 600-s rest, eight SOC steps with 20 A pulses (one logging
    gap, one 5 A charge pulse), 1 A adjustments of 400 s and 300-s rests."""
    T, C = [], []
    clock = 0.0
    for dur, cur in [(3000, 1.5), (600, 0.0)]:
        t, i, clock = step(clock, dur, cur)
        T.append(t)
        C.append(i)
    for k in range(8):
        pulse = 5.0 if k == 5 else -20.0
        for dur, cur, gap in [(10, pulse, 9.778 if k == 3 else None), (0.04, 0.0, None), (400, -1.0, None), (300, 0.0, None)]:
            if dur < 1:
                t = clock + np.array([0.033, 0.043])
                i = np.zeros(2)
                clock += 0.043
            else:
                t, i, clock = step(clock, dur, cur, gap)
            T.append(t)
            C.append(i)
    return np.concatenate(T), np.concatenate(C)


def simulate_truth(t, I, ocv0, kappa, r0, R, tau):
    x = np.zeros(len(tau))
    Q = 0.0
    V = np.empty(len(t))
    for k in range(len(t)):
        if k:
            dt = max(t[k] - t[k - 1], 0.0)
            a = np.exp(-dt / tau)
            x = a * x + (1 - a) * I[k]
            Q += I[k] * dt / 3600.0
        V[k] = ocv0 + kappa * Q + r0 * I[k] + np.dot(R, x)
    return V, x, Q


def identify(t, I, V, n_rc, lam, kap, return_bank=False, lams=(0.999,)):
    rd = M.Reader({'syn': dict(t=t, I=I, V=V)})
    out = A.run_pass(rd, 'syn', 0, [(len(t), 'end')], n_rc, lams, return_bank=return_bank)[len(t)]
    cfg = [(l_, k_) for k_ in (False, True) for l_ in lams]
    return out[cfg.index((lam, kap))]


def sop(E, r0, R, tau, z, K, discharge, steps=801):
    return float(A.rc_power_nk(np.array([E]), np.array([r0]), np.array([R]), np.array([tau]), np.array([z]),
                               np.array([K]), np.array([discharge]), np.array([10.0]), steps)[0])


def closed_form(t, I, V, tau, lam, kap, n_rc):
    """Unrolled stabilised forgetting: R_N = prod(lam) R_p + sum_j prod_{i>j}(lam_i) [(1-lam_j) R_p + phi_j phi_j^T]."""
    x = np.zeros(len(tau))
    Q = 0.0
    t_load = -np.inf
    tp = None
    Phi, Y, LK = [], [], []
    for k in range(len(t)):
        if tp is None:
            dt = 0.0
        else:
            dt = max(t[k] - tp, 0.0)
            a = np.exp(-dt / tau)
            x = a * x + (1 - a) * I[k]
            Q += I[k] * dt / 3600.0
        tp = t[k]
        if abs(I[k]) >= M.REST_A:
            t_load = t[k]
        if not (abs(I[k]) >= M.REST_A or t[k] - t_load <= M.RELAX_S):
            continue
        Phi.append(np.r_[1.0, [Q] if kap else [], I[k], x])
        Y.append(V[k])
        LK.append(lam ** dt)
    Phi, Y, LK = np.array(Phi), np.array(Y), np.array(LK)
    w = np.r_[np.cumprod(LK[::-1])[::-1][1:], 1.0]
    p0 = np.array([M.P0_E] + ([A.P0_KAPPA] if kap else []) + [M.P0_R] * (1 + n_rc))
    Rp = np.diag(1 / p0)
    th0 = np.zeros(len(p0))
    th0[0] = V[0]
    c = np.prod(LK) + np.sum(w * (1 - LK))
    Rn = c * Rp + (Phi * w[:, None]).T @ Phi
    bn = c * (Rp @ th0) + (Phi * w[:, None]).T @ Y
    return np.linalg.solve(Rn, bn)


def independent_power_nk(E, r0, R, tau, z0, K, discharge, h):
    sign = -1 if discharge else 1

    def trial(power):
        def calc(y):
            z, q = y[:-1], y[-1]
            u = E + K * q + np.sum(z, axis=0)
            disc = u ** 2 + 4 * sign * r0 * power
            v = .5 * (u + np.sqrt(np.maximum(disc, 0)))
            return v, power / np.maximum(v, 1e-12), disc

        def rhs(_, y):
            _, current, _ = calc(y)
            return np.r_[(sign * R * current - y[:-1]) / tau, sign * current / 3600.0]
        sol = solve_ivp(rhs, (0, h), np.r_[z0, 0.0], rtol=1e-9, atol=1e-11, dense_output=True, method='LSODA')
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


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-12)))


def identification_checks():
    rng = np.random.default_rng(SEED)
    truths = {1: dict(r0=0.020, R=np.array([0.015]), j=np.array([10])),
              2: dict(r0=0.020, R=np.array([0.010, 0.025]), j=np.array([7, 15]))}
    res = {}
    for prof_name, prof, ocv0, kap_true in [('no_drift', synthetic_profile, 3.7, 0.0), ('ocv_drift', drift_profile, 3.45, 0.35)]:
        t, I = prof()
        for n_rc, tr in truths.items():
            tau = M.TAU_GRID[tr['j']]
            V, x_end, Q_end = simulate_truth(t, I, ocv0, kap_true, tr['r0'], tr['R'], tau)
            E_end = ocv0 + kap_true * Q_end
            true_dis = sop(E_end, tr['r0'], tr['R'], tau, tr['R'] * x_end, kap_true, True)
            true_chg = sop(E_end, tr['r0'], tr['R'], tau, tr['R'] * x_end, kap_true, False)
            for noise in (0.0, 0.001):
                Vn = V + (rng.normal(0, noise, len(V)) if noise else 0.0)
                for kap in (False, True):
                    s = identify(t, I, Vn, n_rc, 0.999, kap)
                    o = dict(status=s['status'])
                    if s['status'] != 'missing':
                        jsel = np.searchsorted(M.TAU_GRID, s['tau'] - 1e-12)
                        dis = sop(s['E_anchor'], s['R0'], s['R'], s['tau'], s['z0'], s['kappa'], True)
                        chg = sop(s['E_anchor'], s['R0'], s['R'], s['tau'], s['z0'], s['kappa'], False)
                        o.update(tau_index=jsel.tolist(), true_tau_index=tr['j'].tolist(), R0_rel_err=abs(s['R0'] / tr['r0'] - 1),
                                 R_rel_err=np.abs(s['R'] / tr['R'] - 1).tolist(), kappa=s['kappa'], kappa_true=kap_true,
                                 E_anchor_err_V=s['E_anchor'] - E_end, sop_dis_W=dis, sop_dis_true_W=true_dis,
                                 sop_chg_W=chg, sop_chg_true_W=true_chg)
                    ok = s['status'] == 'ok'
                    # tolerances (stated in PROTOCOL_AMENDMENTS.md); the structure without kappa on drifting data is
                    # reported without a tolerance
                    if prof_name == 'ocv_drift' and not kap:
                        o['tolerance'] = 'none (structure without kappa on drifting data, reported only)'
                        o['passed'] = True
                    elif noise == 0.0:
                        rtol = 1e-3 if prof_name == 'no_drift' else 1e-2
                        o['passed'] = bool(ok and np.array_equal(jsel, tr['j']) and o['R0_rel_err'] <= rtol
                                           and max(o['R_rel_err']) <= rtol and abs(o['E_anchor_err_V']) <= 1e-3
                                           and abs(dis - true_dis) <= 0.05 * max(1, true_dis / 10)
                                           and abs(chg - true_chg) <= 0.05 * max(1, true_chg / 10)
                                           and (abs(s['kappa'] - kap_true) <= (0.01 if kap_true == 0 else 0.05 * kap_true)))
                    else:
                        o['passed'] = bool(ok and np.all(np.abs(jsel - tr['j']) <= 1) and o['R0_rel_err'] <= 0.02
                                           and max(o['R_rel_err']) <= 0.15 and abs(dis / true_dis - 1) <= 0.02
                                           and abs(chg / true_chg - 1) <= 0.02
                                           and (abs(s['kappa'] - kap_true) <= (0.05 if kap_true == 0 else 0.2 * kap_true)))
                    res[f'{prof_name}/{n_rc}RC/noise_{noise * 1000:g}mV/kappa_{"on" if kap else "off"}'] = o
            # recursion equals the unrolled closed form; stacked configs equal single-lambda banks
            Vn = V + rng.normal(0, 0.001, len(V))
            idx = M.member_taus(n_rc)
            m_true = int(np.flatnonzero((idx == tr['j']).all(1))[0])
            for kap in (False, True):
                s_all = identify(t, I, Vn, n_rc, 0.998, kap, return_bank=True, lams=M.LAMBDAS)
                s_one = identify(t, I, Vn, n_rc, 0.998, kap, return_bank=True, lams=(0.998,))
                cf = closed_form(t, I, Vn, tau, 0.998, kap, n_rc)
                res[f'{prof_name}/{n_rc}RC/recursion_vs_closed_form/kappa_{"on" if kap else "off"}'] = dict(
                    max_rel_diff=rel(s_all['theta_bank'][m_true], cf),
                    stacked_vs_single_max_abs=float(np.max(np.abs(s_all['theta_bank'] - s_one['theta_bank']))),
                    passed=bool(rel(s_all['theta_bank'][m_true], cf) <= 1e-8
                                and np.max(np.abs(s_all['theta_bank'] - s_one['theta_bank'])) <= 1e-12))
    return res


def solver_checks():
    rng = np.random.default_rng(SEED)
    n = 2000
    E = rng.uniform(3.0, 4.15, n)
    r0 = rng.uniform(0.005, 0.08, n)
    R = np.stack([rng.uniform(0.001, 0.05, n), rng.uniform(0.001, 0.05, n)], 1)
    tau = np.sort(10 ** rng.uniform(-1, 3, (n, 2)), 1)
    z = rng.uniform(-0.05, 0.05, (n, 2))
    dis = rng.random(n) < 0.5
    h = np.where(rng.random(n) < 0.5, 10.0, 2.0)
    a = M.rc_power_n(E, r0, R, tau, z, dis, h, 201)
    b = A.rc_power_nk(E, r0, R, tau, z, np.zeros(n), dis, h, 201)
    m = 20
    K = rng.uniform(0, 3, m)
    p = A.rc_power_nk(E[:m], r0[:m], R[:m], tau[:m], z[:m], K, dis[:m], np.full(m, 10.0), 801)
    ind = np.array([independent_power_nk(E[i], r0[i], R[i], tau[i], z[i], K[i], dis[i], 10.0) for i in range(m)])
    return dict(kappa0_equals_rc_power_n=dict(cases=n, max_abs_diff_W=float(np.nanmax(np.abs(a - b))),
                                              passed=bool(np.nanmax(np.abs(a - b)) == 0)),
                independent_ode_with_kappa=dict(cases=m, max_abs_diff_W=float(np.max(np.abs(p - ind))),
                                                passed=bool(np.max(np.abs(p - ind)) <= 0.05)))


def main():
    res = dict(identification=identification_checks(), solver=solver_checks())
    res['all_passed'] = bool(all(v['passed'] for v in res['identification'].values())
                             and all(v['passed'] for v in res['solver'].values()))
    (EXP / 'results').mkdir(exist_ok=True)
    (EXP / 'results/checks_synthetic_a1.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
