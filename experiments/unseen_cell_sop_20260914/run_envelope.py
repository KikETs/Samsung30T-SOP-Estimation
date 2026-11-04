"""Nested cell evaluation; writes only under this experiment directory."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from envelope import Envelope, select, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT/'analysis'
sys.path.insert(0, str(ANALYSIS))
from temp_defects import defective_hppc


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read_labels(direction):
    path = ANALYSIS/('sop_label_charge.csv' if direction=='charge' else 'sop_label_measured.csv')
    bad = defective_hppc()
    rows, excluded = [], {}
    for idx, raw in enumerate(csv.DictReader(path.open())):
        cell, cycle = raw['cell'], int(raw['cycle'])
        reason = None
        ysigned = float(raw['I_star_lin4_A'])
        v, limit = float(raw['V_pre_V']), float(raw['V_min_V'])
        ex, tau = float(raw['extrap']), float(raw['tau_s'])
        if (cell, cycle) in bad:
            reason = 'existing_temperature_fault'
        elif not all(np.isfinite([ysigned, v, limit, ex, tau])):
            reason = 'nonfinite_reference_or_input'
        elif abs(ysigned)<=0.5 or (ysigned<=0 if direction=='charge' else ysigned>=0):
            reason = 'reference_sign_or_min_current'
        if reason:
            excluded[reason] = excluded.get(reason, 0)+1
            continue
        rows.append(dict(row_id=f'{direction}:{idx}', cell=cell, cycle=cycle,
                         direction=direction, tau=tau, v=v, limit=limit,
                         headroom=(limit-v if direction=='charge' else v-limit),
                         y=abs(ysigned), extrap=ex, soc=float(raw['SOC']),
                         soh=float(raw['SOH']), max_measured_A=abs(float(raw['I_max_meas_A']))))
    return rows, dict(path=str(path), sha256=digest(path), excluded=excluded,
                      available_rows=len(rows), primary_rows=sum(r['extrap']<=1.5 for r in rows))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    out=HERE/'results_envelope_v1'
    out.mkdir(exist_ok=True)
    protocol_hash=digest(HERE/'PROTOCOL.md')
    sources={}; summaries=[]; predictions=[]; checks=[]
    for direction, tol in [('discharge', 0.), ('charge', .5)]:
        allrows, sources[direction]=read_labels(direction)
        cells=sorted({r['cell'] for r in allrows})
        for tau in [2.,10.]:
            task=[r for r in allrows if r['tau']==tau]
            primary=[r for r in task if r['extrap']<=1.5]
            for outer in cells:
                tr=[r for r in primary if r['cell']!=outer]
                conf, ranking=select(tr, tol)
                model=Envelope(tr, conf)
                tag=f'{direction}_{int(tau)}s_{outer}'
                # Persist model and inner selection before accessing test targets.
                state=model.state()
                assert outer not in state['train_cells']
                dump(out/'models'/f'{tag}.json', dict(outer=outer, **state,
                     train_row_ids=[r['row_id'] for r in tr], protocol_sha256=protocol_hash))
                dump(out/'selection'/f'{tag}.json', ranking)
                te=[r for r in task if r['cell']==outer]
                pred, upper, fallback=model.predict([r['v'] for r in te], [r['headroom'] for r in te])
                for r,p,rr,fb in zip(te,pred,upper,fallback):
                    predictions.append(dict(**r, predicted_A=float(p), resistance_upper_ohm=float(rr),
                        nearest_voltage_fallback=bool(fb), tolerance_A=tol,
                        over_A=float(p-r['y']), model=tag))
                for stratum, mask in [('primary', np.array([r['extrap']<=1.5 for r in te])),
                                     ('interpolation', np.array([r['extrap']<=1 for r in te])),
                                     ('all_labels', np.ones(len(te), dtype=bool))]:
                    if not mask.any():
                        continue
                    rr=[r for r,m in zip(te,mask) if m]
                    s=dict(direction=direction,tau_s=tau,cell=outer,stratum=stratum,
                           tolerance_A=tol,**conf.record(), **metrics(rr,pred[mask],tol),
                           fallback_count=int(fallback[mask].sum()),
                           inner_selection_safe=ranking[0]['inner_exceed_count']==0)
                    summaries.append(s)
                    if stratum=='primary':
                        print(tag, 'n',s['n'],'exceed',s['exceed_count'],
                              'worst',round(s['worst_A'],3),'usable',round(s['median_usable_pct'],1),
                              conf.record(), flush=True)
                # A test-target mutation cannot affect predictions: predict only accepts V/headroom.
                before=model.state()
                poisoned=[dict(r,y=r['y']*1000,soh=-10,cell='POISON') for r in te]
                again=model.predict([r['v'] for r in poisoned],[r['headroom'] for r in poisoned])[0]
                checks.append(dict(model=tag,disjoint_cells=outer not in model.cells,
                                   finite_predictions=bool(np.isfinite(pred).all()),
                                   target_mutation_invariant=bool(np.array_equal(again,pred)),
                                   state_unchanged=before==model.state()))
    write_csv(out/'predictions.csv', predictions)
    write_csv(out/'summary.csv', summaries)
    primary=[s for s in summaries if s['stratum']=='primary']
    gate=dict(outer_groups=len(primary), exceed_count=sum(s['exceed_count'] for s in primary),
              minimum_cell_task_median_usable_pct=min(s['median_usable_pct'] for s in primary),
              zero_output_count=sum(s['zero_output_count'] for s in primary),
              safety_pass=all(s['exceed_count']==0 for s in primary),
              utility_pass=all(s['median_usable_pct']>=50 for s in primary),
              physical_unseen_cell_guarantee=False,
              reason='Retrospective label-based experiment; external and physical validation remain open.')
    dump(out/'verification.json',dict(checks=checks,gate=gate))
    dump(out/'sources.json', dict(sources=sources,protocol_sha256=protocol_hash,
          scripts={p.name:digest(p) for p in [HERE/'envelope.py', HERE/'run_envelope.py']},
          input_hashes_unchanged=all(digest(Path(s['path']))==s['sha256'] for s in sources.values())))
    print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    main()
