"""Audit data-to-plot mapping, unclipped glyphs, font embedding and export bounds."""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import fitz
from PIL import Image
import cairosvg

OUT=Path(__file__).resolve().parents[1]
p=pd.read_csv(OUT/'Fig3_plot_data.csv')
g=json.loads((OUT/'geometry_Fig3.json').read_text(encoding='utf-8'))
provenance=json.loads((OUT/'provenance_Fig3.json').read_text(encoding='utf-8'))
assert len(p)==30 and p.groupby('arm').size().eq(15).all()
assert p.groupby('arm').n.sum().eq(672).all()
assert p.groupby('arm').exceedances.sum().to_dict()=={'Ls B0':7,'Ls+AD':28}
assert p.groupby('arm').passed.sum().to_dict()=={'Ls B0':8,'Ls+AD':11}
assert p.groupby('arm').strict.sum().to_dict()=={'Ls B0':6,'Ls+AD':4}
assert p.overflow.sum()==2 and np.allclose(p.loc[p.overflow,'exceedance_pct'],200/3)
assert p.median_utilisation_pct.between(70,101).all()
assert p.loc[~p.overflow,'exceedance_pct'].between(0,30).all()
assert np.allclose(p.exceedance_pct,100*p.exceedances/p.n)
for arm,offset in [('Ls B0',-.12),('Ls+AD',.12)]:
    d=p[p.arm==arm];assert np.allclose(d.display_y-d.row_center,offset)

doc=fitz.open(OUT/'Fig3.pdf');page=doc[0]
assert np.allclose([page.rect.width*25.4/72,page.rect.height*25.4/72],[190,125],atol=.001)
assert not page.get_images()
fonts=[]
for f in page.get_fonts():
    name,ext,kind,blob=doc.extract_font(f[0]);assert blob
    fonts.append({'name':name,'embedded_bytes':len(blob)})
spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
assert min(s['size'] for s in spans)>=6.99
assert all(s['size']>=7.99 or s['text'].strip() in ['Joint','Strict'] for s in spans)
assert all(page.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
fw,fh=g['figure_pixels'][2:]
for t in g['texts']:
    x,y,w,h=t['bbox_px']
    assert x>=0 and y>=0 and x+w<=fw+.01 and y+h<=fh+.01,t['text']
    assert t['size_pt']>=(7 if t['text'] in ['Joint','Strict'] else 8)
alltext=' '.join(t['text'] for t in g['texts'])
assert all(s in alltext for s in ['8/15','6/15','11/15','4/15','−20','−10'])
assert '66.7' not in alltext
for key,center_mm in [('methods',99),('verdicts',170)]:
    x,y,w,h=g['legend_bounds_px'][key]
    assert abs((x+w/2)/fw*190-center_mm)<.01
assert g['legend_bounds_px']['methods'][0]+g['legend_bounds_px']['methods'][2]<g['legend_bounds_px']['verdicts'][0]
assert not any(s in alltext.lower() for s in ['intervention','int_rate'])
assert sum(t['text']=='Joint' for t in g['texts'])==2
assert 'Util. &' not in alltext
for label in ['Fixed','Online','above 30%','Pass','Fail']:
    assert any(t['text']==label and (1-(t['bbox_px'][1]+t['bbox_px'][3]/2)/fh)*125>116
               for t in g['texts']),label
direction_texts=[t for t in g['texts'] if t['text'] in ['Charge','Discharge']]
temp_texts=[t for t in g['texts'] if t['text'].endswith(' °C')]
count_texts=[t for t in g['texts'] if t['text'].startswith('(') and t['text'][1:-1].isdigit()]
assert [len(direction_texts),len(temp_texts),len(count_texts)]==[5,15,15]
for labels,right_mm in [(direction_texts,28.5),(temp_texts,39.5),(count_texts,51.8)]:
    assert all(abs((t['bbox_px'][0]+t['bbox_px'][2])/fw*190-right_mm)<.01 for t in labels)
for t,group in zip(direction_texts,g['direction_groups']):
    assert t['size_pt']==10
    assert abs((1-(t['bbox_px'][1]+t['bbox_px'][3]/2)/fh)*125-group['center_page_y_mm'])<.01

# Compare marker centers/radii against the true axes rectangle including padding.
for name,metric,lo,hi in [('a','median_utilisation_pct',70,101),('b','exceedance_pct',-1,31)]:
    ax_left,ax_bot,ax_w,ax_h=g['axes'][name]
    for _,r in p.iterrows():
        value=min(r[metric],30) if name=='b' else r[metric]
        radius_pt=(4.3 if name=='b' and r.overflow else 4)/2+.5
        radius_mm=radius_pt*25.4/72
        x_mm=(value-lo)/(hi-lo)*ax_w*190
        y_mm=(r.display_y+.6)/(15.9+.6)*ax_h*125
        assert radius_mm<=x_mm<=ax_w*190-radius_mm,(name,r.row_index,'x')
        assert radius_mm<=y_mm<=ax_h*125-radius_mm,(name,r.row_index,'y')

ns={'s':'http://www.w3.org/2000/svg'};svg=ET.parse(OUT/'Fig3.svg')
assert not svg.findall('.//s:image',ns)
assert len(svg.findall('.//s:text',ns))>=40
png=Image.open(OUT/'Fig3.png')
assert png.size==(4488,2952) and abs(png.info['dpi'][0]-600)<.01
cairosvg.svg2png(url=str(OUT/'Fig3.svg'),write_to=str(OUT/'Fig3_svg_render.png'),output_width=1197)
# High-resolution details for the tightly spaced marks and two-level verdict key.
for name,box in [('B_rows',(0,62,190,102)),('verdicts',(149,1,190,125))]:
    page.get_pixmap(clip=fitz.Rect(*(v*72/25.4 for v in box)),dpi=300,alpha=False).save(OUT/f'Fig3_detail_{name}.png')
result={'status':'PASS','source_summary':provenance['summary'],'dimensions_mm':[190,125],
 'minimum_font_pt':min(s['size'] for s in spans),'fonts':fonts,'pdf_bitmap_images':0,
 'svg_live_text_nodes':len(svg.findall('.//s:text',ns)),'png_pixels':list(png.size),
 'overflow_points':2,'overflow_true_pct':200/3,'marker_geometry':'All markers fit axes bounds, including explicitly labelled overflow triangles.',
 'data_offsets':'Only perpendicular +/-0.12 row offsets; no value-axis jitter.',
 'label_alignment':'5 centered 10 pt direction labels; 15 independently right-aligned temperature and count labels.'}
(OUT/'verification_Fig3.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
files=['Fig3.pdf','Fig3.svg','Fig3.png','Fig3_plot_data.csv','caption_Fig3.md',
       'source/build_fig3.py','source/verify_fig3.py','source/style.py','provenance_Fig3.json']
(OUT/'sha256_Fig3.json').write_text(json.dumps({f:hashlib.sha256((OUT/f).read_bytes()).hexdigest()
    for f in files if (OUT/f).exists()},indent=2),encoding='utf-8')
print('PASS: data, alignment, export bounds, embedded fonts, 7 pt Joint/Strict keys and >=8 pt remaining text.')
