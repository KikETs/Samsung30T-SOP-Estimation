"""Independent row, frontier, table and vector-output checks for Fig. 5."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import fitz,cairosvg
from PIL import Image

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
p=pd.read_csv(OUT/'Fig5_plot_data.csv');prov=json.loads((OUT/'provenance_Fig5.json').read_text(encoding='utf-8'))
g=json.loads((OUT/'geometry_Fig5.json').read_text(encoding='utf-8'))
cols=['tau','exceed_rows','int_rate','util_q50','closed_loop']
loc=pd.read_csv(ROOT/'paper_package_lsad_20261013_r3/data/iso_tradeoff_posthoc.csv')
pd.testing.assert_frame_equal(pd.read_csv(OUT/'Fig5_local_plot_data.csv'),loc,rtol=1e-14,atol=1e-14)
assert len(p)==811
maps=[('pulse','method',{'PERSISTENCE':'Previous pulse power','RINT_PREVIOUS':'Previous pulse resistance'}),
      ('circuit','candidate',{'ECM-1RC-RLS-A1':'Equivalent circuit, one RC branch','ECM-2RC-RLS-A1':'Equivalent circuit, two RC branches'})]
for src,key,names in maps:
    raw=pd.read_csv(prov['sources'][src]['path'])
    for code,name in names.items():
        for rawpol,pol in [('fixed','Fixed'),('AD','Online')]:
            expected=raw[raw[key].eq(code)&raw.policy.eq(rawpol)].sort_values('tau')[cols].reset_index(drop=True)
            actual=p[p.method.eq(name)&p.scale.eq(pol)].sort_values('tau')[cols].reset_index(drop=True)
            assert len(actual)==81
            pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-14,atol=1e-14)
            assert np.allclose(actual.tau,np.arange(81)/400,rtol=0,atol=1e-14)
local=p[p.method.eq('Five local predictors')]
for pol,rawpol in [('Fixed','fixed scale'),('Online','online adaptation (tau* = tau)')]:
    pd.testing.assert_frame_equal(local[local.scale.eq(pol)][cols].reset_index(drop=True),
                                 loc[loc.policy.eq(rawpol)][cols].reset_index(drop=True),check_dtype=False,rtol=1e-14,atol=1e-14)
for fn,subset in [('Fig5_fixed_frontier.csv',local[local.scale.eq('Fixed')]),('Fig5_local_envelope.csv',local)]:
    frontier=pd.read_csv(OUT/fn)
    for col in ['util_q50','closed_loop']:
        assert np.allclose(frontier[col],[subset.loc[subset.exceed_rows.le(n),col].max() for n in frontier.exceed_rows],rtol=0,atol=1e-14)
comp=p[~p.method.isin(['Five local predictors','Reproduction of Chen et al.'])]
assert len(comp)==648
for col in ['util_q50','closed_loop']:
    assert all(r[col]<=local.loc[local.exceed_rows.le(r.exceed_rows),col].max()+1e-12 for _,r in comp.iterrows())
tab=pd.read_csv(prov['sources']['table']['path'])
for (method,policy),d in p.groupby(['method','scale']):
    t=tab[tab.predictor.str.startswith(method)].iloc[0];r=d[np.isclose(d.tau,.05)].iloc[0]
    prefix='fixed' if policy=='Fixed' else 'adapted'
    assert int(r.exceed_rows)==int(t[prefix+'_exceedances'])
    assert int(r.closed_loop)==int(t[prefix+'_util_exc_groups'])
    assert abs(r.util_q50*100-t[prefix+'_median_utilisation_pct'])<.000051
for name,a in g['axes'].items():
    values=p if name in ['(c)','(d)'] else local
    assert values.int_rate.mul(100).between(*a['xlim']).all()
    y=values.util_q50*100 if name in ['(a)','(c)'] else values.closed_loop
    assert y.between(*a['ylim']).all()
assert g['axes']['(a)']['ylim']==g['axes']['(c)']['ylim']
assert g['axes']['(b)']['ylim']==g['axes']['(d)']['ylim']
assert len({tuple(a['xlim']) for a in g['axes'].values()})==1
fw,fh=g['figure_pixels'][2:]
for t in g['texts']:
    x,y,w,h=t['bbox_px'];assert min(x,y)>=-.01 and x+w<=fw+.01 and y+h<=fh+.01,t['text']
    assert t['size_pt']>=8
doc=fitz.open(OUT/'Fig5.pdf');page=doc[0]
assert np.allclose([page.rect.width*25.4/72,page.rect.height*25.4/72],[190,170],atol=.001)
assert not page.get_images()
fonts=[]
for fnt in page.get_fonts():
    name,ext,kind,blob=doc.extract_font(fnt[0]);assert blob;fonts.append({'name':name,'bytes':len(blob)})
spans=[s for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
assert min(s['size'] for s in spans)>=7.99
assert all(page.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
pdftext=page.get_text()
for forbidden in ['PERSISTENCE','RINT_','ECM-','tau*','AD']:
    assert forbidden not in pdftext,forbidden
ns={'s':'http://www.w3.org/2000/svg'};svg=ET.parse(OUT/'Fig5.svg')
assert not svg.findall('.//s:image',ns)
assert len(svg.findall('.//s:text',ns))>=60
png=Image.open(OUT/'Fig5.png');assert png.size==(4488,4015)
assert abs(png.info['dpi'][0]-600)<.01
cairosvg.svg2png(url=str(OUT/'Fig5.svg'),write_to=str(OUT/'Fig5_svg_render.png'),output_width=1197)
result={'status':'PASS','plot_records':811,'local_records':162,'comparator_curve_records':648,'single_point_records':1,
 'points_above_local_envelope':{'utilisation':0,'passing_groups':0},
 'all_observations_inside_axes':True,
 'dimensions_mm':[190,170],'png_pixels':list(png.size),
 'pdf_images':0,'pdf_embedded_fonts':fonts,'svg_text_nodes':len(svg.findall('.//s:text',ns))}
(OUT/'verification_Fig5.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
files=[f for f in OUT.iterdir() if f.is_file() and f.name!='sha256_Fig5.json']
files+=list((OUT/'source').glob('*.py'))
(OUT/'sha256_Fig5.json').write_text(json.dumps({str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print('PASS:648 comparator curve rows match sources,162 local rows unchanged,Chen single point correct,all Table5 checks pass,no clipped observations,vector exports verified.')
