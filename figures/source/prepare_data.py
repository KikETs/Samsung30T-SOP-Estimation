"""Extract original observations and check the excerpt against the actual predictor inputs."""
from pathlib import Path
import pandas as pd
import numpy as np
import json, hashlib

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
DATA = OUT/'data'
DATA.mkdir(exist_ok=True)
raw = next((OUT/'raw').glob('*.CSV'))
d = pd.read_csv(raw, low_memory=False)
d.insert(0, 'file_row', np.arange(len(d)))
hpath = DATA/'history_A1_excerpt.csv'
h = pd.read_csv(hpath)
aligned = d.iloc[h.file_row]
mapping = {'time_s':'Test_Time(s)', 'current_A':'Current(A)', 'voltage_V':'Voltage(V)',
           'temperature_C':'Aux_Temperature_3(C)'}
history_errors = {f'max_abs_{k}':float(np.max(abs(aligned[v].to_numpy()-h[k].to_numpy())))
                  for k,v in mapping.items() if k!='time_s'}
history_errors['max_abs_relative_time_s'] = float(np.max(abs(
    aligned['Test_Time(s)'].to_numpy()-aligned['Test_Time(s)'].iloc[0]-h.time_s.to_numpy())))
assert len(h)==910 and history_errors['max_abs_voltage_V']<=.0001 and history_errors['max_abs_current_A']<=.001
featurespath = PROJECT/'experiments/unseen_cell_sop_20260914/posthoc_lsad_v320/feats_ref_fixed.csv'
f = pd.read_csv(featurespath).set_index('row_id').loc['rpcwby:Test#1:1219']
reqpath = PROJECT/'paper_package_lsad_20261013_r3/data/ad_trajectory.csv'
req = pd.read_csv(reqpath).set_index('row_id').loc['rpcwby:Test#1:1219']
# Counter 2 identifies the SOC set point, counter 1 identifies repeated sweeps.
# Step 87 applies the constant-power discharge at SOC 0.1 in each CP sweep.
trials=[]
for sweep in range(2,7):
    q=d[(d.Step_Index==87)&(d.TC_Counter1==sweep)&(d.TC_Counter2==12)].copy()
    assert len(q)>0 and q['Current(A)'].lt(0).all()
    q['trial']=sweep-1
    q['pulse_time_s']=q['Step_Time(s)']
    q['power_magnitude_W']=abs(q['Voltage(V)']*q['Current(A)'])
    # Keep observations at <=10 s. The duplicate termination record at 10.001 s
    # remains in the complete raw timeline and is separately audited below.
    trials.append(q[q.pulse_time_s<=10.0])
search=pd.concat(trials)
search.to_csv(DATA/'search_pulses.csv',index=False)
final=trials[-1]
start=float(final['Test_Time(s)'].iloc[0]-final['Step_Time(s)'].iloc[0])
end=float(final['Test_Time(s)'].iloc[-1])
cutoffrow=int(final.file_row.iloc[0]-1)
cutoff=float(d.iloc[cutoffrow]['Test_Time(s)'])
window=d[(d['Test_Time(s)']>=cutoff-900)&(d['Test_Time(s)']<=end+30)].copy()
window['query_time_s']=window['Test_Time(s)']-start
window['power_magnitude_W']=abs(window['Voltage(V)']*window['Current(A)'])
window.to_csv(DATA/'query_window.csv',index=False)
prev=d[(d.Step_Index==86)&(d.TC_Counter1==6)&(d.TC_Counter2==11)&(d['Step_Time(s)']<=10.0)].copy()
prev_end=prev.iloc[-1]
assert abs(abs(prev_end['Current(A)']*prev_end['Voltage(V)'])-f.prev_P)<1e-7
lastload=int(prev.file_row.iloc[0])-1
rest_start=lastload
while rest_start>0 and abs(d.iloc[rest_start-1]['Current(A)'])<.01:
    rest_start-=1
pre_rest=float(d.iloc[lastload]['Test_Time(s)']-d.iloc[rest_start]['Test_Time(s)'])
final_V=float(final['Voltage(V)'].iloc[-1]); final_P=float(final.power_magnitude_W.iloc[-1])
assert abs(final_V-2.5593)<=.0005 and abs(final_V-2.55)<=.04
assert abs(final_P-65.6)<=.05
meta={
    'row_id':'rpcwby:Test#1:1219', 'request':req.to_dict(),
    'column_mapping':mapping, 'temperature_note':'Mapping reproduces the legacy excerpt exactly. Sensor interpretation is not needed for these voltage/power plots.',
    'original_excerpt':{'samples':910, **history_errors, 'first_absolute_time_s':float(aligned['Test_Time(s)'].iloc[0]),
       'last_absolute_time_s':float(aligned['Test_Time(s)'].iloc[-1]),
       'gap_to_final_pulse_start_s':start-float(aligned['Test_Time(s)'].iloc[-1])},
    'query':{'start_absolute_time_s':start,'cutoff_file_row':cutoffrow,'cutoff_absolute_time_s':cutoff,
       'end_absolute_time_s':end,'end_file_row':int(final.file_row.iloc[-1]),
       'nominal_duration_s':10.0,'last_step_record_s':10.001,'end_voltage_V':final_V,
       'end_power_VI_W':final_P, 'end_power_recorded_W':float(abs(final['Power(W)'].iloc[-1])),
       'limit_V':2.55,'limit_error_mV':(final_V-2.55)*1000},
    'preceding_pulse':{'start_absolute_time_s':float(prev['Test_Time(s)'].iloc[0]-prev['Step_Time(s)'].iloc[0]),
       'first_measured_absolute_time_s':float(prev['Test_Time(s)'].iloc[0]),
       'end_absolute_time_s':float(prev_end['Test_Time(s)']), 'nominal_duration_s':10.0,
       'end_power_W':float(f.prev_P),'rest_before_pulse_s':pre_rest,
       'feature_rest_before_pulse_s':float(f.prev_rest_s), 'feature_query_rest_s':float(f.rest_s)},
    'trials':[{'trial':i+1,'sweep_counter':i+2,'start_absolute_time_s':float(q['Test_Time(s)'].iloc[0]-q.pulse_time_s.iloc[0]),
       'power_label_W':round(float(q.power_magnitude_W.iloc[-1]),1),
       'end_voltage_V':float(q['Voltage(V)'].iloc[-1]),'samples_in_0_to_10_s':len(q)} for i,q in enumerate(trials)],
    'excerpt_discrepancies':[
      'The supplied 910-sample excerpt precedes the final query by 78283.096 s, not immediately before the final query.',
      'Same-SOC search trials are separated by complete sweeps, with other loads and recharges between them.',
      'The supplied excerpt contains no qualifying constant-power preceding pulse. The predictor input is the 84.201393 W pulse in the final sweep.',
      'The raw protocol has a repeated termination record at 10.001 s. Nominal 10 s and 10.001 s records have identical endpoint voltage and power.'
    ],
    'sources':[{'path':str(p),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in [raw,hpath,featurespath,reqpath]],
}
(OUT/'data_audit.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:meta[k] for k in ['original_excerpt','query','preceding_pulse','trials']},indent=2))
