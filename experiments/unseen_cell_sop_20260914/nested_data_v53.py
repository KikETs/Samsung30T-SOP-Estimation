"""Explicit fit-set preparation for the unchanged V52 method."""
from collections import Counter
from itertools import combinations
import numpy as np
from run_envelope import HERE, read_labels
from waveform_forward_v20 import read_csv
from multitemperature_data_v50 import load_data

CELLS = ('BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2')
MENDELEY = 'MENDELEY_SN001'

def fit_sets():
    return [tuple(c) for n in (4, 5) for c in combinations(CELLS, n)]

def validate_cells(fit_cells):
    chosen = tuple(sorted(fit_cells))
    if len(chosen) not in (4, 5) or len(set(chosen)) != len(chosen) or not set(chosen) <= set(CELLS):
        raise ValueError('Expected four or five distinct UYPYDJ fit cells')
    return chosen

def train_indices(rows, direction, fit_cells):
    chosen = set(validate_cells(fit_cells)) | {MENDELEY}
    return np.array([j for j, r in enumerate(rows) if r['direction'] == direction and r['cell'] in chosen])

def fit_weights(rows, indices, fit_cells):
    chosen = validate_cells(fit_cells)
    cell_names = np.array([rows[j]['cell'] for j in indices])
    counts = Counter(cell_names)
    assert set(counts) == set(chosen) | {MENDELEY}
    temps = Counter(int(rows[j]['nominal_temperature_C']) for j in indices if rows[j]['cell'] == MENDELEY)
    assert set(temps) == {-20, -10, 0, 10, 25, 40}
    ncell = len(counts)
    result = []
    for j in indices:
        r = rows[j]
        denominator = ncell * 6 * temps[int(r['nominal_temperature_C'])] if r['cell'] == MENDELEY else ncell * counts[r['cell']]
        result.append(len(indices) / denominator)
    return np.asarray(result, dtype=np.float32)

def cap_from_training(rows, current, lengths, direction, fit_cells):
    chosen = set(validate_cells(fit_cells))
    idx = np.array([j for j, r in enumerate(rows) if r['direction'] == direction and r['cell'] in chosen])
    return float(1.5 * np.max(current[idx, lengths[idx]-1]))

def upper_targets(rows, current, lengths, fit_cells):
    chosen = validate_cells(fit_cells)
    byid = {r['row_id']: j for j, r in enumerate(rows)}
    links = {r['row_id']: r['source_response_row_id'] for r in read_csv(HERE/'horizon_response_data_v1/label_query_links.csv')}
    ref = np.zeros(len(rows), np.float32)
    horizon = np.ones(len(rows), np.float32)
    weight = np.zeros(len(rows), np.float32)
    records, sources = [], {}
    for direction in ('discharge', 'charge'):
        labels, source = read_labels(direction)
        sources[source['path']] = source['sha256']
        cap = cap_from_training(rows, current, lengths, direction, chosen)
        labels = [r for r in labels if r['cell'] in chosen and r['extrap'] <= 1.5]
        counts = Counter(r['cell'] for r in labels if r['y'] < cap)
        assert set(counts) == set(chosen)
        old_n = sum(r['direction'] == direction and r['cell'] in chosen for r in rows)
        new_n = len(train_indices(rows, direction, chosen))
        # V52 derives these counts with sum(numpy_bool_array), yielding a
        # numpy.float64 ratio. Preserve that promotion before float32 storage.
        factor = np.float64(new_n) / np.float64(old_n)
        for r in labels:
            j = byid[links[r['row_id']]]
            assert rows[j]['cell'] == r['cell'] and rows[j]['direction'] == direction and ref[j] == 0
            ref[j], horizon[j] = r['y'], r['tau']
            active = r['y'] < cap
            original = old_n / (len(chosen) * counts[r['cell']]) if active else 0.
            # Match V52's float32 assignment followed by float32 multiplication.
            weight[j] = np.float32(original)
            records.append(dict(row_id=r['row_id'], source_response_row_id=rows[j]['row_id'],
                cell=r['cell'], direction=direction, reference_A=r['y'], horizon_s=r['tau'],
                cap_A=cap, active=active, original_v43_weight=original, weight=original*factor))
        direction_mask = np.array([r['direction'] == direction for r in rows])
        weight[direction_mask] *= factor
    return ref, horizon, weight, records, sources

def local_neighbors(rows, train, neighbors):
    local = {rows[j]['row_id']: k for k, j in enumerate(train)}
    index = np.full(len(train), -1, np.int64)
    weight = np.zeros(len(train), np.float32)
    for k, j in enumerate(train):
        r = rows[j]
        if r['row_id'] not in neighbors:
            continue
        pair = neighbors[r['row_id']]
        other = pair['neighbor_id']
        if other not in local:
            raise ValueError('Neighbor outside explicit fit set')
        target = rows[train[local[other]]]
        assert r['cell'] == target['cell'] and r['direction'] == target['direction']
        assert r['source_file'] == target['source_file'] and r['clock_segment'] == target['clock_segment']
        assert float(r['requested_horizon_s']) == float(target['requested_horizon_s'])
        index[k] = local[other]
        weight[k] = np.exp(-float(pair['state_distance'])/2)
    return index, weight
