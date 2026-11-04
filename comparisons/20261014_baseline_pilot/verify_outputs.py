"""Independent score recount; does not import the predictor or evaluator implementation."""
from pathlib import Path
import csv, json, hashlib, statistics
from collections import defaultdict

root=Path(__file__).resolve().parent
out=root/'results'
def read(name):
    with (out/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

requests=read('requests.csv'); reported=read('primary.csv'); group_report=read('groups.csv')
with (root/'sources/closedloop_v251/src/bench_rows.csv').open(encoding='utf-8-sig') as f:
    keys={r['row_id']:(r['evaluation_group'],r['direction'],float(r['temp_C']),float(r['horizon_s'])) for r in csv.DictReader(f)}
errors=[];arms=[]
for report in reported:
    method,policy=report['method'],report['policy']
    rr=[r for r in requests if r['method']==method and r['policy']==policy]
    assert len(rr)==len({r['row_id'] for r in rr})==672
    assert {r['row_id'] for r in rr}==set(keys)
    grouped=defaultdict(list);ratios=[]
    for r in rr:
        ratio=float(r['request'])/float(r['truth']);ratios.append(ratio);grouped[keys[r['row_id']]].append(ratio)
    exceed=sum(r>1+1e-9 for r in ratios)
    median=statistics.median(min(r,1) for r in ratios)
    passed=strict=0
    for key,a in grouped.items():
        e=sum(r>1+1e-9 for r in a);u=statistics.median(min(r,1) for r in a)
        passed+=u>=.8 and e/len(a)<=.1
        strict+=e==0 and statistics.median(a)>=.8
    actual={'exceed_rows':exceed,'util_q50':median,'closed_loop':passed,'original':strict}
    for field,v in actual.items():
        if abs(v-float(report[field]))>1e-12:errors.append([method,policy,field,v,report[field]])
    arms.append({'method':method,'policy':policy,**actual})
hashes=json.loads((out/'output_hashes.json').read_text())
for name,expected in hashes.items():
    if hashlib.sha256((out/name).read_bytes()).hexdigest()!=expected:errors.append(['hash',name])
result={'arms':len(arms),'request_rows':len(requests),'score_recount_errors':errors,'verified_output_hashes':len(hashes),'scores':arms}
(out/'independent_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2));assert not errors
