"""All stationary points of two exponentials plus signed-square charge.

Each q-sign segment has V''' equal to two exponentials, with at most
one zero. Recursively partition V'', then V', to enumerate every extremum.
"""
import numpy as np
from scipy.optimize import brentq


def _roots(function, boundaries):
    roots = []
    boundaries = sorted(set(boundaries))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        a, b = function(left), function(right)
        if a == 0:
            roots.append(left)
        if b == 0:
            roots.append(right)
        if a != 0 and b != 0 and np.signbit(a) != np.signbit(b):
            roots.append(brentq(function, left, right, xtol=1e-13))
    return roots


def _segment(parameters, states, q0, current, duration, pre):
    r0, r1, t1, p1, r2, t2, p2, k, b = parameters
    s1, s2 = states
    a1 = (r1 * current - s1) / t1
    a2 = (r2 * current - s2) / t2

    def value(t):
        q = q0 + current * t
        return (r0 * (current - pre) + s1 * np.exp(-t / t1)
                + r1 * current * (-np.expm1(-t / t1)) - p1
                + s2 * np.exp(-t / t2) + r2 * current * (-np.expm1(-t / t2)) - p2
                + k * q + b * q * abs(q))

    def first(t):
        return (a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)
                + k * current + 2 * b * abs(q0 + current * t) * current)

    pieces = [0., float(duration)]
    if current != 0 and 0 < -q0 / current < duration:
        pieces.append(float(-q0 / current))
    pieces.sort()
    candidates = list(pieces)
    # V''' = a1/t1^2 exp(-t/t1) + a2/t2^2 exp(-t/t2).
    turning = None
    if a1 * a2 < 0 and t1 != t2:
        turning = ((np.log(abs(a1)) - 2 * np.log(t1)
                    - np.log(abs(a2)) + 2 * np.log(t2)) / (1 / t1 - 1 / t2))
    for left, right in zip(pieces[:-1], pieces[1:]):
        sign = np.sign(q0 + current * ((left + right) / 2))

        def second(t):
            return (-a1 / t1 * np.exp(-t / t1) - a2 / t2 * np.exp(-t / t2)
                    + 2 * b * current * current * sign)

        bounds = [left, right]
        if turning is not None and left < turning < right:
            bounds.append(float(turning))
        critical = _roots(second, bounds)
        first_bounds = sorted(set([left, right] + critical))
        candidates.extend(critical)  # retain possible tangent zeros as candidates
        candidates.extend(_roots(first, first_bounds))
    values = [value(t) for t in candidates]
    j = int(np.argmax(values))
    return float(values[j]), float(candidates[j])


def maximum_response(parameters, pre, gap, horizon, current):
    p = parameters
    s1 = p[3] * np.exp(-gap / p[2]) + p[1] * pre * (-np.expm1(-gap / p[2]))
    s2 = p[6] * np.exp(-gap / p[5]) + p[4] * pre * (-np.expm1(-gap / p[5]))
    future = _segment(p, (s1, s2), pre * gap, current, horizon, pre)
    previous = _segment(p, (p[3], p[6]), 0., pre, gap, pre)
    # Report time relative to the future current switch; negative means gap.
    previous = (previous[0], previous[1] - gap)
    return max([(0., -gap), previous, future], key=lambda item: item[0])


def invert(parameters, pre, gap, horizon, headroom, cap, margin=0.):
    p = np.asarray(parameters, float)
    if p.ndim != 2 or p.shape[1] != 9 or not np.isfinite(p).all():
        raise ValueError('Invalid parameters')
    if (p[:, [0, 1, 2, 4, 5]] <= 0).any() or (p[:, 7:] < 0).any():
        raise ValueError('Nonpassive parameters')
    arrays = [np.broadcast_to(np.asarray(v, float), (len(p),))
              for v in [pre, gap, horizon, headroom]]
    pre, gap, horizon, headroom = arrays
    if (not all(np.isfinite(a).all() for a in arrays) or (gap < 0).any()
            or (horizon <= 0).any() or not np.isfinite(cap) or cap < 0
            or not np.isfinite(margin) or margin < 0):
        raise ValueError('Invalid state or limits')
    currents, times, bads = [], [], []
    for params, ii, g, h, head in zip(p, pre, gap, horizon, headroom):
        target = head - .020 - margin

        def residual(j):
            return maximum_response(params, ii, g, h, j)[0] - target

        bad = residual(0.) > 0
        current = 0. if bad else cap if residual(cap) <= 0 else brentq(residual, 0., cap, xtol=1e-12)
        currents.append(current)
        times.append(maximum_response(params, ii, g, h, current)[1])
        bads.append(bad)
    return np.array(currents), np.array(times), np.array(bads)
