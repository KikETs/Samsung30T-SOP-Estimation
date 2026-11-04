"""Independent voltage-input SOP ceiling. No oracle state or test-label query."""
from dataclasses import dataclass
import itertools
import numpy as np

WIDTHS = (0.025, 0.05, 0.10, 0.20, float('inf'))
GAPS = (0., 1., 2.)
RESERVES = (1., 1.025, 1.05, 1.10, 1.20, 1.40, 1.60, 2.)
V_GUARD = 0.020
EPS = 1e-9


@dataclass(frozen=True)
class Config:
    width: float
    gap: float
    reserve: float

    def record(self):
        return dict(width_V=self.width if np.isfinite(self.width) else 'global',
                    gap_multiplier=self.gap, reserve=self.reserve)


def grid():
    return [Config(*x) for x in itertools.product(WIDTHS, GAPS, RESERVES)]


class Envelope:
    def __init__(self, rows, config):
        self.config = config
        self.cells = sorted({r['cell'] for r in rows})
        if len(self.cells) < 2:
            raise ValueError('At least two independent training cells required')
        self.data = {}
        for cell in self.cells:
            rr = [r for r in rows if r['cell'] == cell and r['headroom'] > 0]
            if not rr:
                raise ValueError(f'No positive-headroom training rows: {cell}')
            self.data[cell] = (np.array([r['v'] for r in rr]),
                               np.array([r['headroom']/r['y'] for r in rr]))

    def predict(self, voltage, headroom):
        # No measured current, SOC, SOH, cycle, or cell ID is accepted here.
        v = np.asarray(voltage, float)
        h = np.asarray(headroom, float)
        if not (np.isfinite(v).all() and np.isfinite(h).all()):
            raise ValueError('Non-finite deployment input')
        maxima = []
        fallback = np.zeros(len(v), dtype=bool)
        for vv, rr in self.data.values():
            out = np.empty(len(v))
            for k in range(0, len(v), 128):
                dist = np.abs(v[k:k+128, None] - vv[None, :])
                mask = dist <= self.config.width + V_GUARD
                empty = ~mask.any(axis=1)
                # Include all ties at the nearest voltage, not an arbitrary row.
                mask[empty] = dist[empty] <= dist[empty].min(axis=1)[:, None]+EPS
                out[k:k+128] = np.where(mask, rr[None, :], -np.inf).max(axis=1)
                fallback[k:k+128] |= empty
            maxima.append(out)
        ordered = np.sort(np.array(maxima), axis=0)
        upper = (ordered[-1] + self.config.gap*(ordered[-1]-ordered[-2]))
        upper *= self.config.reserve
        if not np.isfinite(upper).all() or (upper <= 0).any():
            raise ValueError('Invalid learned resistance ceiling')
        return np.maximum(h-V_GUARD, 0)/upper, upper, fallback

    def state(self):
        return dict(config=self.config.record(), train_cells=self.cells,
                    data={c: {'voltage': v.tolist(), 'slope': r.tolist()}
                          for c, (v, r) in self.data.items()})


def metrics(rows, pred, tolerance):
    y = np.array([r['y'] for r in rows])
    over = np.asarray(pred)-y
    ratio = np.asarray(pred)/y
    return dict(n=len(y), exceed_count=int((over>tolerance+EPS).sum()),
                strict_exceed_count=int((over>EPS).sum()),
                worst_A=float(max(over.max(), 0)),
                worst_beyond_tolerance_A=float(max(over.max()-tolerance, 0)),
                median_usable_pct=float(np.median(ratio)*100),
                p05_usable_pct=float(np.percentile(ratio, 5)*100),
                min_usable_pct=float(ratio.min()*100),
                zero_output_count=int((np.asarray(pred)<=EPS).sum()))


def select(training, tolerance):
    """All inputs belong to the OUTER training cells; no global OOF table."""
    cells = sorted({r['cell'] for r in training})
    configs = grid()
    evaluations = [[] for _ in configs]
    # Compute each neighbourhood just once per inner fold/width/gap.
    for val in cells:
        tr = [r for r in training if r['cell'] != val]
        te = [r for r in training if r['cell'] == val]
        vv, hh = [r['v'] for r in te], [r['headroom'] for r in te]
        cache = {}
        for i, conf in enumerate(configs):
            key = (conf.width, conf.gap)
            if key not in cache:
                model = Envelope(tr, Config(conf.width, conf.gap, 1.))
                cache[key] = model.predict(vv, hh)[0]
            p = cache[key]/conf.reserve
            evaluations[i].append(dict(cell=val, **metrics(te, p, tolerance)))
    rankings = []
    for conf, ev in zip(configs, evaluations):
        count = sum(e['exceed_count'] for e in ev)
        worst = max(e['worst_beyond_tolerance_A'] for e in ev)
        utility = min(e['median_usable_pct'] for e in ev)
        mean_utility = float(np.mean([e['median_usable_pct'] for e in ev]))
        # Failure first, then worst excess (only for failed candidates), utility.
        rank = (count>0, worst if count else 0, -utility, -mean_utility)
        rankings.append((rank, conf, dict(**conf.record(),
                          inner_exceed_count=count, inner_worst_excess_A=worst,
                          inner_min_cell_median_usable_pct=utility,
                          inner_mean_cell_median_usable_pct=mean_utility,
                          folds=ev)))
    rankings.sort(key=lambda r: r[0])
    return rankings[0][1], [r[2] for r in rankings]
