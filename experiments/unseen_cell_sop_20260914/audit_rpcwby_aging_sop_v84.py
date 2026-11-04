"""V84: RPCWBY Test#1/#2 (Samsung 30T aging cells) measured 10 s SOP - definitions, raw linkage, provenance.

Source alignment only; no model is evaluated here. Pulses are identified from the Arbin step structure and the
protocol counters (TC_Counter1 = pass, TC_Counter2 = SOC point, TC_Counter4 = block) before any table value is read.
Constant-power (CP) apply pulses are the iterative SOP search; CCCV pulses (pass 1) give the 'SOP_*_CCCV' seed value.
The workbook is parsed with a block parser that keeps annotated cycle labels (e.g. 'Cycle 1187 (...)').
"""
import configparser,hashlib,io,json,re,sys,time,zipfile
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'rpcwby_aging_sop_v84'
ZIPS={'Test#1':ROOT/'raw/RPCWBY/1_Test_1.zip','Test#2':ROOT/'raw/RPCWBY/2_Test_2.zip'}
XLSX=ROOT/'raw/RPCWBY/0_SOP_summary.xlsx';PROTO=ROOT/'raw/RPCWBY/Arbin Cycler Protocols.zip'
SOC_POINT={1:1.0,2:.95,3:.9,4:.8,5:.7,6:.6,7:.5,8:.4,9:.3,10:.2,11:.15,12:.1,13:.05,14:.02,15:.005}
UD_FOR_POINT={1:'UD2',2:'UD3',3:'UD4',4:'UD5',5:'UD6',6:'UD7',7:'UD8',8:'UD9',9:'UD10',10:'UD11',11:'UD12',12:'UD13',13:'UD14',14:'UD15',15:'UD1'}
LIMITS=dict(discharge=dict(current_A=30.,voltage_V=2.55),charge=dict(current_A=15.,voltage_V=4.15))
TOL=dict(current_A=.1,voltage_V=.04);HORIZON_S=10.
USECOLS=['Data_Point','Date_Time','Test_Time(s)','Step_Time(s)','Cycle_Index','Step_Index','TC_Counter1','TC_Counter2','TC_Counter3','TC_Counter4','Current(A)','Voltage(V)','Power(W)','Charge_Capacity(Ah)','Discharge_Capacity(Ah)']

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def protocol_definitions():
    out=[]
    with zipfile.ZipFile(PROTO) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('.sdx') and not n.startswith('__MACOSX')):
            blob=z.read(name);c=configparser.ConfigParser(interpolation=None,strict=False);c.optionxform=str;c.read_string(blob.decode('latin1'))
            steps=[];formulas={}
            for s in c.sections():
                d=dict(c[s])
                if re.fullmatch(r'Schedule_Step\d+',s):
                    lims=[]
                    for ls in c.sections():
                        if ls.startswith(s+'_Limit'):
                            L=dict(c[ls]);lims.append(dict(test=' & '.join(f"{L.get(f'Equation{k}_szLeft','')}{L.get(f'Equation{k}_szCompareSign','')}{L.get(f'Equation{k}_szRight','')}" for k in range(3) if L.get(f'Equation{k}_szLeft','')),goto=L.get('m_szGotoStep','')))
                    steps.append(dict(step=int(s.removeprefix('Schedule_Step')),label=d.get('m_szLabel',''),control=d.get('m_szStepCtrlType',''),value=d.get('m_szCtrlValue',''),extra=d.get('m_szExtCtrlValue1',''),limits=lims))
                elif 'Formula' in s:formulas[d.get('m_szLabel',s)]=d.get('m_szExpression','')
            out.append(dict(member=name,sha256=sha(blob),steps=steps,formulas=formulas,
                cp_apply_steps=[s['step'] for s in steps if s['control']=='Power(W)' and 'apply' in s['label']],
                cccv_pulse_steps=[s['step'] for s in steps if s['control']=='CCCV' and 'Pulse' in s['label']]))
    return out

def parse_workbook():
    wb=openpyxl.load_workbook(XLSX,data_only=True,read_only=True);rows=[];caps=[]
    for sheet in ['Test#1','Test#2']:
        grid=[list(r) for r in wb[sheet].iter_rows(values_only=True)]
        starts=[i for i,r in enumerate(grid) if len(r)>1 and isinstance(r[1],str) and re.match(r'\s*Cycle\s*\d+',r[1])]
        for bi,i in enumerate(starts):
            label=grid[i][1].strip();cycle=int(re.search(r'\d+',label).group());stop=starts[bi+1] if bi+1<len(starts) else len(grid)
            temps={};capnote={}
            for j in range(i,min(i+5,stop)):
                for col,v in enumerate(grid[j][:9]):
                    if isinstance(v,str):
                        m=re.match(r'\s*(-?\d+)\s*°?\s*C',v)
                        if m:
                            temps[col]=int(m.group(1));mc=re.search(r'(\d\.\d{3,})',v)
                            if mc:capnote[col]=float(mc.group(1))
            header_row=next((j for j in range(i,min(i+6,stop)) if any(isinstance(v,str) and v.startswith('SOP_disch') for v in grid[j][:9])),None)
            headers={col:grid[header_row][col] for col in range(1,9) if header_row is not None and isinstance(grid[header_row][col],str)}
            for j in range(i,stop):
                soc=grid[j][0]
                if not isinstance(soc,(int,float)) or not (0<=soc<=1):continue
                for col in sorted(temps):
                    vals={}
                    for k,field in enumerate(['SOP_disch','SOP_disch_CCCV','SOP_char','SOP_char_CCCV']):
                        v=grid[j][col+k] if col+k<len(grid[j]) else None
                        vals[field]=float(v) if isinstance(v,(int,float)) else None
                    if any(v is not None for v in vals.values()):
                        rows.append(dict(sheet=sheet,cycle=cycle,cycle_label=label,block_row=i+1,sheet_row=j+1,temp_C=temps[col],SOC=round(float(soc),4),
                            first_col=col+1,disch_header=headers.get(col,''),cap_note_Ah=capnote.get(col),**vals))
        # capacity side table: 'Cycle#' / 'Capacity (Ah)'
        for i,r in enumerate(grid):
            for col,v in enumerate(r):
                if v=='Cycle#' and col+1<len(r) and isinstance(r[col+1],str) and r[col+1].startswith('Capacity'):
                    for j in range(i+1,len(grid)):
                        c=grid[j][col] if col<len(grid[j]) else None;a=grid[j][col+1] if col+1<len(grid[j]) else None
                        if isinstance(c,(int,float)) and isinstance(a,(int,float)):caps.append(dict(sheet=sheet,cycle=int(c),capacity_Ah=float(a),note=grid[j][col+2] if col+2<len(grid[j]) and isinstance(grid[j][col+2],str) else ''))
                        elif c is None and a is None:break
    return pd.DataFrame(rows),pd.DataFrame(caps)

def read_member(z,name):
    d=pd.read_csv(io.BytesIO(z.read(name)),low_memory=False)
    d.columns=[c.replace('﻿','').strip() for c in d.columns]
    aux=[c for c in d.columns if c.startswith('Aux_Temperature')];ud=[c for c in d.columns if c.startswith('MetaCode_MV_UD')]
    for c in ud:d[c]=pd.to_numeric(d[c],errors='coerce')
    return d,aux,ud

def scan_file(sheet,name,d,aux,ud):
    tc=aux[0] if aux else None;ts=aux[1] if len(aux)>1 else None
    t=d['Test_Time(s)'].to_numpy(float);I=d['Current(A)'].to_numpy(float);V=d['Voltage(V)'].to_numpy(float);P=d['Power(W)'].to_numpy(float);st=d['Step_Time(s)'].to_numpy(float)
    change=np.flatnonzero(d.Step_Index.ne(d.Step_Index.shift()).to_numpy());ends=np.r_[change[1:]-1,len(d)-1]
    udlog={}
    for c in ud:
        s=d[c].dropna()
        if len(s):udlog[c.replace('MetaCode_MV_','')]=(s.index.to_numpy(),s.to_numpy(float))
    pulses=[]
    for k,(a,b) in enumerate(zip(change,ends)):
        dur=st[b];seg_I=I[a:b+1]
        if not (abs(dur-HORIZON_S)<.15 and np.abs(seg_I).max()>.1):continue
        prange=float(np.ptp(P[a:b+1]));irange=float(np.ptp(seg_I));direction='charge' if I[b]>0 else 'discharge'
        lim=LIMITS[direction];c1,c2,c3,c4=[int(d[f'TC_Counter{n}'].iloc[b]) for n in (1,2,3,4)]
        kind=('hppc_or_other' if c2<1 else 'cccv_seed' if c1==1 else 'constant_power' if prange<.5 else 'nonconstant_power_apply')
        nxt_a,nxt_b=(change[k+1],ends[k+1]) if k+1<len(change) else (b,b)
        ud16=d['MetaCode_MV_UD16'].iloc[nxt_a:nxt_b+1].dropna() if 'MetaCode_MV_UD16' in d else pd.Series(dtype=float)
        setpoint=None;var=UD_FOR_POINT.get(c2)
        if var in udlog:
            idx,vals=udlog[var];prior=idx<a
            if prior.any():setpoint=float(vals[prior][-1])
        rest=a-1
        while rest>0 and abs(I[rest])>.05:rest-=1
        t0=float(t[a]-st[a])
        pulses.append(dict(sheet=sheet,member=name,first_row=int(a),last_row=int(b),n_samples=int(b-a+1),step_index=int(d.Step_Index.iloc[a]),cycle_index=int(d.Cycle_Index.iloc[a]),
            date_time=str(d.Date_Time.iloc[a]),start_time_s=t0,first_sample_lag_s=float(st[a]),duration_s=float(dur),kind=kind,direction=direction,
            pass_counter=c1,soc_counter=c2,unconverged_counter=c3,block_counter=c4,soc_point=SOC_POINT.get(c2),
            I_first_A=float(seg_I[0]),I_last_A=float(I[b]),V_first_V=float(V[a]),V_last_V=float(V[b]),P_first_W=float(P[a]),P_last_W=float(P[b]),power_range_W=prange,current_range_A=irange,
            max_abs_current_A=float(np.abs(seg_I).max()),min_voltage_V=float(V[a:b+1].min()),max_voltage_V=float(V[a:b+1].max()),
            chamber_temp_C=float(np.median(d[tc].iloc[a:b+1])) if tc else None,surface_temp_C=float(np.median(d[ts].iloc[a:b+1])) if ts else None,
            calculated_sop_after_W=float(ud16.iloc[-1]) if len(ud16) else None,setpoint_W=setpoint,
            end_current_error_A=float(abs(abs(I[b])-lim['current_A'])),end_voltage_error_V=float(abs(V[b]-lim['voltage_V'])),
            prior_row=int(a-1),prior_voltage_V=float(V[a-1]),prior_current_A=float(I[a-1]),prior_time_s=float(t[a-1]),rest_before_s=float(t[a]-t[rest]) if rest>=0 else None,
            available_history_s=float(t0-t[0]),discharge_capacity_Ah=float(d['Discharge_Capacity(Ah)'].iloc[a]),charge_capacity_Ah=float(d['Charge_Capacity(Ah)'].iloc[a])))
    info=dict(sheet=sheet,member=name,rows=len(d),first_date=str(d.Date_Time.iloc[0]),last_date=str(d.Date_Time.iloc[-1]),span_h=float((t[-1]-t[0])/3600),
              chamber_col=tc,surface_col=ts,chamber_median_C=float(d[tc].median()) if tc else None,chamber_min_C=float(d[tc].min()) if tc else None,chamber_max_C=float(d[tc].max()) if tc else None,
              nonincreasing_time_rows=int((np.diff(t)<=0).sum()),ten_second_pulses=len(pulses))
    return pulses,info

def main():
    assert not OUT.exists(),'Inspect existing output before rerunning';OUT.mkdir()
    started=time.monotonic();sources={str(p):sha(p.read_bytes()) for p in [XLSX,PROTO,Path(__file__)]}
    proto=protocol_definitions();dump(OUT/'protocol_definitions.json',proto)
    table,caps=parse_workbook();table.to_csv(OUT/'table_values.csv',index=False);caps.to_csv(OUT/'table_capacity.csv',index=False)
    files=[];pulses=[]
    for sheet,zp in ZIPS.items():
        sources[str(zp)]=None
        with zipfile.ZipFile(zp) as z:
            names=sorted(n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX'))
            for name in names:
                d,aux,ud=read_member(z,name);p,info=scan_file(sheet,name,d,aux,ud);info['sha256']=sha(z.read(name))
                pulses+=p;files.append(info);print(sheet,name.split('/')[-1],'rows',len(d),'pulses',len(p),'elapsed',round(time.monotonic()-started),flush=True)
    for zp in ZIPS.values():
        h=hashlib.sha256()
        with zp.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<24),b''):h.update(chunk)
        sources[str(zp)]=h.hexdigest()
    P=pd.DataFrame(pulses);F=pd.DataFrame(files)
    P['temp_block_C']=np.where((P.chamber_temp_C-25).abs()<=4,25,np.where((P.chamber_temp_C-10).abs()<=4,10,-999))
    P['date']=pd.to_datetime(P.date_time,format='%m/%d/%Y %H:%M:%S.%f',errors='coerce')
    P=P.sort_values(['sheet','date','first_row']).reset_index(drop=True);P['pulse_id']=np.arange(len(P))
    P.to_csv(OUT/'pulses.csv',index=False);F.to_csv(OUT/'files.csv',index=False)
    # final constant-power pulse per (sheet, member, temp block, direction, soc point): highest pass then latest row
    cp=P[(P.kind=='constant_power')&(P.soc_counter>=1)&(P.temp_block_C>0)].copy()
    P['converged']=(P.end_current_error_A<=TOL['current_A'])|(P.end_voltage_error_V<=TOL['voltage_V'])
    cp['final']=False
    key=['sheet','member','temp_block_C','direction','soc_counter']
    last=cp.sort_values(['pass_counter','first_row']).groupby(key).tail(1).index;cp.loc[last,'final']=True
    cccv=P[(P.kind=='cccv_seed')&(P.temp_block_C>0)]
    finals=cp[cp.final].copy()
    finals['cccv_seed_W']=np.nan
    for i,r in finals.iterrows():
        q=cccv[(cccv.sheet==r.sheet)&(cccv.member==r.member)&(cccv.temp_block_C==r.temp_block_C)&(cccv.direction==r.direction)&(cccv.soc_counter==r.soc_counter)]
        if len(q):finals.at[i,'cccv_seed_W']=float(q.iloc[-1].P_last_W)
    # link table values: pass 1 finds exact measured-power matches anywhere in the sheet/temp/direction/SOC;
    # the dominant member per (sheet,cycle,temp) defines the cycle->file map; pass 2 restricts candidates to that file.
    finals['converged']=(finals.end_current_error_A<=TOL['current_A'])|(finals.end_voltage_error_V<=TOL['voltage_V'])
    def candidates(r,direction,member=None):
        c=finals[(finals.sheet==r.sheet)&(finals.temp_block_C==r.temp_C)&(finals.direction==direction)&(np.isclose(finals.soc_point,r.SOC,atol=1e-6))]
        return c if member is None else c[c.member==member]
    def value_status(cand,val,cval,tol):
        exact=cand[(cand.P_last_W-val).abs()<=tol]
        if len(exact)==1:return 'measured_final_power',exact.iloc[0]
        if len(exact)>1:return 'ambiguous_'+str(len(exact)),None
        seed=cand[(cand.setpoint_W.fillna(np.inf)-abs(val)).abs()<=tol]
        if len(seed)==1:return 'setpoint_value',seed.iloc[0]
        if cval is not None and not (isinstance(cval,float) and np.isnan(cval)):
            cs=cand[(cand.cccv_seed_W.fillna(np.inf)-abs(cval)).abs()<=tol]
            if len(cs)==1 and abs(val-cval)<=tol:return 'cccv_seed_value_in_sop_column',cs.iloc[0]
        return 'unmatched',None
    FIELDS=[('discharge','SOP_disch','SOP_disch_CCCV'),('charge','SOP_char','SOP_char_CCCV')]
    votes={}
    for r in table.itertuples():
        for direction,field,cfield in FIELDS:
            val=getattr(r,field)
            if val is None or (isinstance(val,float) and np.isnan(val)):continue
            cand=candidates(r,direction);tol=max(.0055,abs(val)*1e-5)
            exact=cand[(cand.P_last_W-val).abs()<=tol]
            for m in exact.member.unique():votes.setdefault((r.sheet,r.cycle,r.temp_C),{}).setdefault(m,0);votes[(r.sheet,r.cycle,r.temp_C)][m]+=1
    cycle_map=[]
    for (sheet,cycle,temp),vs in sorted(votes.items()):
        ranked=sorted(vs.items(),key=lambda kv:-kv[1]);member=ranked[0][0]
        cycle_map.append(dict(sheet=sheet,cycle=cycle,temp_C=temp,member=member,exact_votes=ranked[0][1],competing_members=len(ranked)-1,competing_votes=sum(v for _,v in ranked[1:]),
            member_first_date=str(F[F.member==member].first_date.iloc[0])))
    CM=pd.DataFrame(cycle_map);CM.to_csv(OUT/'cycle_file_map.csv',index=False)
    mapped={(r.sheet,r.cycle,r.temp_C):r.member for r in CM.itertuples()}
    links=[]
    for r in table.itertuples():
        for direction,field,cfield in FIELDS:
            val=getattr(r,field)
            if val is None or (isinstance(val,float) and np.isnan(val)):continue
            member=mapped.get((r.sheet,r.cycle,r.temp_C));cand=candidates(r,direction,member);tol=max(.0055,abs(val)*1e-5)
            cval=getattr(r,cfield);status,v=value_status(cand,val,cval,tol)
            link=dict(sheet=r.sheet,cycle=r.cycle,cycle_label=r.cycle_label,temp_C=r.temp_C,SOC=r.SOC,direction=direction,table_field=field,table_value_W=val,table_cccv_W=cval,
                      mapped_member=member,candidate_finals=len(cand),status=status if member else 'no_file_for_cycle',pulse_id=None,raw_final_power_W=None,delta_W=None,raw_setpoint_W=None,raw_cccv_seed_W=None,
                      nearest_final_power_W=float(cand.iloc[int((cand.P_last_W-val).abs().argmin())].P_last_W) if len(cand) else None,converged=None,end_current_error_A=None,end_voltage_error_V=None,pass_counter=None)
            if v is not None:
                link.update(pulse_id=int(v.pulse_id),raw_final_power_W=float(v.P_last_W),delta_W=float(v.P_last_W-val),raw_setpoint_W=v.setpoint_W,raw_cccv_seed_W=v.cccv_seed_W,
                            converged=bool(v.converged),end_current_error_A=float(v.end_current_error_A),end_voltage_error_V=float(v.end_voltage_error_V),pass_counter=int(v.pass_counter))
            links.append(link)
    L=pd.DataFrame(links);L.to_csv(OUT/'table_links.csv',index=False);finals.to_csv(OUT/'final_pulses.csv',index=False)
    summary=dict(files=len(F),pulses_10s=len(P),constant_power_pulses=int((P.kind=='constant_power').sum()),cccv_seed_pulses=int((P.kind=='cccv_seed').sum()),hppc_or_other_pulses=int((P.kind=='hppc_or_other').sum()),nonconstant_power_applies=int((P.kind=='nonconstant_power_apply').sum()),
        final_cp_pulses=len(finals),final_cp_converged=int(finals.converged.sum()),table_values=len(L),link_status=L.status.value_counts().to_dict(),cycle_file_map_rows=len(CM),cycle_file_map_competing=int((CM.competing_votes>0).sum()),
        final_pulse_temp_blocks=finals.groupby(['sheet','temp_block_C','direction']).size().reset_index().values.tolist(),
        definitions=dict(unit='W',sign='discharge negative, charge positive (workbook); raw Power(W) signed the same',control='constant power (Arbin Power(W) step) for the iterative SOP; CCCV pulse (CC at current limit then CV at voltage limit) for *_CCCV',
            duration_s=HORIZON_S,limits=LIMITS,convergence_tolerance=TOL,convergence_rule='pass ends when |I_end|-I_limit within 0.1 A or |V_end-V_limit| within 0.04 V (LS_CHAN last-step values); otherwise SOP updated by AF_SOP formula and re-applied in the next pass',
            early_stop='discharge pulse aborted if V<2.5 V before 10 s (Counter3++); charge uses OCV>4.15 skip',soc_axis='protocol F_SOC = (discharged Ah - charged Ah)/capacity since 1C CCCV charge to 4.2 V; capacity = last 1C capacity test (agingSOP_1) or constant 2.1274 Ah (agingsop_2_CC)',
            measured_or_computed='table SOP = power measured at the last sample of the final applied constant-power pulse (matches raw), *_CCCV = |V*I| at the end of the CCCV pulse; both are cycler measurements, the CP search value is the 10 s constant-power capability',
            history='SOP measured after 1C (1.5 A) CCCV charge to 4.2 V, 15 min rest, stepwise 1C discharge with 5 min rests; multiple passes revisit SOC points'),
        elapsed_s=time.monotonic()-started,goal_complete=False,model_evaluated=False)
    dump(OUT/'summary.json',summary)
    dump(OUT/'audit.json',dict(source_hashes=sources,output_hashes={p.name:sha(p.read_bytes()) for p in sorted(OUT.iterdir()) if p.is_file()},goal_complete=False))
    print(json.dumps({k:summary[k] for k in ['files','pulses_10s','constant_power_pulses','cccv_seed_pulses','hppc_or_other_pulses','nonconstant_power_applies','final_cp_pulses','final_cp_converged','table_values','link_status','cycle_file_map_rows','cycle_file_map_competing']},indent=1,ensure_ascii=False))
if __name__=='__main__':main()
