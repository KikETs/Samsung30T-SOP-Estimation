"""Stand-ins for two modules whose import chains read result files deleted in the 2026-10-20 cleanup. Neither is used by
the Ls rebuild: closedloop_v253.simulate is only called by closedloop_v255.deliver (delivery simulation), cross_sop_v199
only inside closedloop_v257 functions (short-rest predictor). Any call raises, so a silent dependency cannot pass."""
import sys
import types


def install():
    for name in ('closedloop_v253', 'cross_sop_v199'):
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__file__ = '<V322 stub>'

        def _unused(*a, _n=name, **k):
            raise RuntimeError(f'{_n} is a V322 stub (not used by the Ls rebuild)')
        for attr in ('simulate', 'judged', 'rows_frame', 'power'):
            setattr(m, attr, _unused)
        sys.modules[name] = m
