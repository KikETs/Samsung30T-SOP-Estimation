"""Automated numerical and export checks of Fig. 1."""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import fitz
from PIL import Image

OUT=Path(__file__).resolve().parents[1]
m=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
raw=pd.read_csv(next((OUT/'raw').glob('*.CSV')),low_memory=False)
win=pd.read_csv(OUT/'data/query_window.csv')
trials=pd.read_csv(OUT/'data/search_pulses.csv')
checks={}
for name,d in [('panel_a',win),('panel_b',trials)]:
    match=raw.iloc[d.file_row.to_numpy()]
    for column in ['Test_Time(s)','Step_Time(s)','Current(A)','Voltage(V)']:
        err=float(np.max(abs(match[column].to_numpy()-d[column].to_numpy())))
        assert err<1e-9
        checks[f'{name}_{column}_max_error']=err
assert len(win)==964
assert sorted(trials.trial.unique())==[1,2,3,4,5]
assert len(trials)==50
checks['panel_a_raw_samples']=len(win)
checks['panel_b_raw_samples']=len(trials)
differences=pd.read_csv(OUT/'data/endpoint_differences.csv')
assert len(differences)==4
for row in differences.itertuples(index=False):
    lv=trials.loc[(trials.trial==row.left_trial)&(trials.pulse_time_s==row.time_s),'Voltage(V)'].item()
    rv=trials.loc[(trials.trial==row.right_trial)&(trials.pulse_time_s==row.time_s),'Voltage(V)'].item()
    assert abs(row.voltage_difference_mV-1000*(lv-rv))<1e-10
checks['endpoint_differences']=differences.to_dict('records')
checks['legacy_excerpt_alignment']=m['original_excerpt']
checks['query']=m['query']
checks['preceding_pulse']=m['preceding_pulse']
checks['request_W']=float(m['request']['request'])
assert abs(m['request']['central']*m['request']['scale']-m['request']['request'])<1e-6
pdf=fitz.open(OUT/'Fig1.pdf');page=pdf[0]
assert abs(page.rect.width*25.4/72-190)<.001
assert abs(page.rect.height*25.4/72-125)<.001
assert not page.get_images()
fonts=[]
for font in page.get_fonts():
    name,ext,typ,content=pdf.extract_font(font[0])
    assert content
    fonts.append({'name':name,'embedded_bytes':len(content)})
checks['embedded_fonts']=fonts
outside=[]
for block in page.get_text('dict')['blocks']:
    for line in block.get('lines',[]):
        for span in line['spans']:
            box=fitz.Rect(span['bbox'])
            if not page.rect.contains(box):outside.append(span['text'])
assert not outside,outside
checks['pdf_text_outside_page']=outside
im=Image.open(OUT/'Fig1.png')
assert im.size==(4488,2952) and abs(im.info['dpi'][0]-600)<.01
checks['png']={'pixels':list(im.size),'dpi':im.info['dpi'],'mode':im.mode}
svg=ET.parse(OUT/'Fig1.svg')
ns={'svg':'http://www.w3.org/2000/svg'}
texts=svg.findall('.//svg:text',ns)
assert len(texts)>30 and len(svg.findall('.//svg:image',ns))==0
checks['svg_text_nodes']=len(texts)
checks['svg_raster_images']=0
checks['status']='PASS numerical and export checks for the corrected figure specification'
(OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
files=['Fig1.pdf','Fig1.svg','Fig1.png','caption_Fig1.md','data/query_window.csv','data/search_pulses.csv','data/endpoint_differences.csv',
       'source/build_fig1.py','source/prepare_data.py','source/style.py','source/fetch_raw.py']
(OUT/'sha256_manifest.json').write_text(json.dumps({f:hashlib.file_digest((OUT/f).open('rb'),'sha256').hexdigest() for f in files},indent=2),encoding='utf-8')
print('PASS: raw observations, endpoint, request arithmetic, PDF size/fonts/vectors, PNG DPI, SVG live text.')
