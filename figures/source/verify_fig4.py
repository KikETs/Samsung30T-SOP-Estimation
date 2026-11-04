"""Validate scientific traces, condition segments and exported artwork."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import fitz
import cairosvg
from PIL import Image

OUT=Path(__file__).resolve().parents[1]
p=pd.read_csv(OUT/'Fig4_plot_data.csv')
g=json.loads((OUT/'geometry_Fig4.json').read_text(encoding='utf-8'))
seg=json.loads((OUT/'Fig4_condition_segments.json').read_text(encoding='utf-8'))
prov=json.loads((OUT/'provenance_Fig4.json').read_text(encoding='utf-8'))
source=pd.read_csv(prov['sources'][0]['path']).sort_values(['cell','order']).reset_index(drop=True)
pd.testing.assert_frame_equal(p,source,check_exact=False,rtol=1e-14,atol=1e-14)
assert p.groupby('cell').size().to_dict()=={'A1':368,'A17':252,'B':52}
assert p.groupby('cell').intervened.sum().to_dict()=={'A1':15,'A17':10,'B':3}
assert p.loc[p.cell.eq('B')&p.intervened,'order'].tolist()==[1,2,7]
for cell,d in p.groupby('cell',sort=False):
    covered=[]
    for s in seg[cell]:
        rows=d[d.order.between(s['start'],s['end'])]
        assert len(rows)==s['end']-s['start']+1
        assert rows.temp_C.eq(s['temp_C']).all() and rows.direction.eq(s['direction']).all()
        assert rows.horizon_s.eq(s['duration_s']).all()
        covered.extend(range(s['start'],s['end']+1))
    assert covered==list(range(len(d)))
    assert prov['summary'][cell]['update_log_residual_max']<2e-10
for panel,cell in [('(a)','A1'),('(b)','A17'),('(c)','B')]:
    assert g['axes'][panel]['ylim']==[.74,.97]
    assert p.loc[p.cell.eq(cell),'scale'].between(.74,.97).all()
assert g['axes']['(c)']['xlim']==g['axes']['(d)']['xlim']
assert np.allclose(g['temperature_zero_rgb'],[1,1,1])
assert [a['order'] for a in g['annotations']]==[0,1,2]
assert [a['label'] for a in g['annotations']]==['0 W','1.258','1.488']

doc=fitz.open(OUT/'Fig4.pdf');page=doc[0]
assert np.allclose([page.rect.width*25.4/72,page.rect.height*25.4/72],[190,180],atol=.001)
assert not page.get_images(), 'Entire master including the temperature bar must remain vector.'
fonts=[]
for f in page.get_fonts():
    name,ext,kind,blob=doc.extract_font(f[0]);assert blob
    fonts.append({'name':name,'embedded_bytes':len(blob)})
spans=[s for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
assert min(s['size'] for s in spans)>=6.99
assert all(page.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
fw,fh=g['figure_pixels'][2:]
for t in g['texts']:
    x,y,w,h=t['bbox_px']
    assert min(x,y)>=0 and x+w<=fw+.01 and y+h<=fh+.01,t['text']
    assert t['size_pt']>=8
ns={'s':'http://www.w3.org/2000/svg'};svg=ET.parse(OUT/'Fig4.svg')
assert not svg.findall('.//s:image',ns)
assert len(svg.findall('.//s:text',ns))>=40
png=Image.open(OUT/'Fig4.png')
assert png.size==(4488,4251) and abs(png.info['dpi'][0]-600)<.01,png.size
cairosvg.svg2png(url=str(OUT/'Fig4.svg'),write_to=str(OUT/'Fig4_svg_render.png'),output_width=1197)
for name,rect in [('B',(15,79,190,160)),('initial_B',(23,80,68,152))]:
    page.get_pixmap(clip=fitz.Rect(*(v*72/25.4 for v in rect)),dpi=300,alpha=False).save(OUT/f'Fig4_detail_{name}.png')
result={'status':'PASS','dimensions_mm':[190,180],'png_pixels':list(png.size),
 'observations':len(p),'exceedances':int(p.intervened.sum()),'cell_summary':prov['summary'],
 'condition_segments':{c:len(v) for c,v in seg.items()},'segment_coverage':'Every query covered exactly once in its cell strip.',
 'fonts':fonts,'minimum_pdf_font_pt':min(s['size'] for s in spans),'normal_text_min_pt':8,
 'pdf_images':0,'svg_live_text_nodes':len(svg.findall('.//s:text',ns)),
 'temperature_zero_rgb':g['temperature_zero_rgb']}
(OUT/'verification_Fig4.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
files=['Fig4.pdf','Fig4.svg','Fig4.png','Fig4_plot_data.csv','Fig4_condition_segments.json',
       'source/build_fig4.py','source/verify_fig4.py','source/style.py','caption_Fig4.md',
       'provenance_Fig4.json']
(OUT/'sha256_Fig4.json').write_text(json.dumps({f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files if (OUT/f).exists()},indent=2),encoding='utf-8')
print('PASS: all 672 source rows, 28 exceedances, step-update timing, condition coverage, common scales, vector PDF/SVG, embedded fonts and 600 dpi PNG.')
