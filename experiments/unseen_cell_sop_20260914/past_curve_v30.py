"""Use only the already completed pulse linked by the existing causal ledger."""
import numpy as np
import torch
from torch import nn
from run_envelope import HERE
from waveform_forward_v20 import STATE_NAMES as OLD_STATE_NAMES
from run_nonlinear_q_pilot_v29 import load_pilot_data as load_base
from nonlinear_q_v29 import Encoder as BaseEncoder, response

FRACTIONS = (.1, .25, .5, .75)
EXTRA_NAMES = [f'past_{h}s_fraction_{f:g}_{field}' for h in (2, 10)
               for f in FRACTIONS for field in ('voltage_change_V', 'current_A')]
STATE_NAMES = OLD_STATE_NAMES + EXTRA_NAMES


def extract_curve(query, source, times, currents, voltages):
    if (query['cell'], query['source_file'], query['clock_segment']) != (source['cell'], source['source_file'], source['clock_segment']):
        raise ValueError('Past pulse belongs to another acquisition stream')
    if source['direction'] != query['direction']:
        raise ValueError('Wrong pulse direction')
    if int(source['endpoint_index']) >= int(query['sample_index']):
        raise ValueError('Past pulse not completed by query sample')
    if float(source['last_access_time_s']) > float(query['pre_voltage_time_s']) + 1e-8:
        raise ValueError('Past pulse not available at pre-voltage decision time')
    age = float(query['start_time_s']) - float(source['last_access_time_s'])
    if not 0 < age <= 7200:
        raise ValueError('Past pulse outside causal history window')
    if (len(times) != len(currents) or len(times) != len(voltages) or len(times) == 0
            or not np.isfinite(times).all() or (np.diff(times) < 0).any()):
        raise ValueError('Invalid past waveform')
    duration = float(source['duration_s'])
    np.testing.assert_allclose(times[-1], duration, atol=1e-7)
    selected = np.maximum(0, np.searchsorted(times, np.array(FRACTIONS)*duration, side='right')-1)
    sign = -1. if source['direction'] == 'discharge' else 1.
    values = np.column_stack([sign*(voltages[selected].astype(float)-float(source['pre_voltage_V'])),
                              sign*currents[selected].astype(float)]).ravel()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite past waveform feature')
    return values, selected


def load_pilot_data():
    rows, x, *rest = load_base()
    with np.load(HERE/'past_curve_data_v30/features.npz') as z:
        assert z['row_ids'].tolist() == [r['row_id'] for r in rows]
        extra = z['features']
    assert x.shape[1] == 47 and extra.shape == (len(rows), 16)
    return rows, np.column_stack([x[:, :45], extra, x[:, 45:]]), *rest


class PastInputLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        state = torch.random.get_rng_state()
        self.extra = nn.Linear(32, 32, bias=False)
        nn.init.zeros_(self.extra.weight)
        torch.random.set_rng_state(state)

    def forward(self, x):
        assert x.shape[1] == 126
        baseline = torch.cat([x[:, :45], x[:, 61:63], x[:, 63:108], x[:, 124:126]], dim=1)
        extra = torch.cat([x[:, 45:61], x[:, 108:124]], dim=1)
        return self.base(baseline) + self.extra(extra)


class Encoder(BaseEncoder):
    def __init__(self, n_features):
        if n_features != 63:
            raise ValueError('Expected 47 existing and 16 past-curve features')
        super().__init__(47)
        self.net[0] = PastInputLinear(self.net[0])
