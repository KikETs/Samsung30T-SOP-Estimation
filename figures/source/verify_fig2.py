"""Verify final figure exports, text geometry and anchored information flow."""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
import fitz
from PIL import Image

OUT=Path(__file__).resolve().parents[1]
g=json.loads((OUT/'geometry_Fig2.json').read_text(encoding='utf-8'))
doc=fitz.open(OUT/'Fig2.pdf');page=doc[0]
dimensions=[page.rect.width*25.4/72,page.rect.height*25.4/72]
assert abs(dimensions[0]-190)<.001 and abs(dimensions[1]-125)<.001
assert not page.get_images()
fonts=[]
for item in page.get_fonts():
    name,extension,kind,blob=doc.extract_font(item[0]);assert blob
    fonts.append({'name':name,'embedded_bytes':len(blob)})
spans=[s for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
# True subscripts/superscripts use 70% of a 10–10.5 pt math base.
# Ordinary labels remain >=8 pt.
assert min(s['size'] for s in spans)>=6.99
assert any('Cambria' in f['name'] for f in fonts)
assert any('Cambria-Italic' in f['name'] for f in fonts)
assert all(page.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
assert min(t['size_pt'] for t in g['texts'])>=8
figw,figh=g['figure_pixels'][2:]
for t in g['texts']:
    x,y,w,h=t['bounds_pixels']
    assert x>=0 and y>=0 and x+w<=figw and y+h<=figh,t['text']
text='\n'.join(t['text'] for t in g['texts'])
required=[r'$e_k=\mathbf{1}[p_k>y_k+10^{-9}\,\mathrm{W}]$',
 r'$u_k=\min[p_k/y_k,\,1]$',
 r'$\log s_{k+1}=\mathrm{clip}[\log s_k+\gamma(\tau^{*}-e_k),$',
 r'$\log 0.20,\,\log 1.50]$',
 r'$\hat{p}_k=\exp[\mathrm{median}_j(\log\hat{p}_j)]$',
 r'$\hat{I}\cdot\hat{R}(\hat{I})=H$',
 r'$\gamma=0.05,\quad\tau^{*}=0.05$',
 r'$s_0=\max\{s\in S:E_{\mathrm{cal}}(s)\leq\tau\}$',
 r'$S=\{0.20,\,0.21,\,\ldots,\,1.50\}$',r'$\tau=5\,\%$',
 r'$x_j,\,H,\,I_{\mathrm{ref}}$',r'$V_{\mathrm{pre}}$',r'$\times\,s_k$']
assert all(s in text for s in required)
assert not any(s in text for s in ['I(ref)','V(pre)','p(k)','τ*','ₚ','₀','⁻⁹'])
# The revised dense equations must fit their own boxes, not merely the page.
for snippet,node in [(r'\hat{p}_k=\exp[','combined'),
                     (r'\log s_{k+1}=','update'),
                     (r'\log 0.20,','update')]:
    matches=[t for t in g['texts'] if snippet in t['text']]
    assert len(matches)==1
    x,y,w,h=matches[0]['bounds_pixels']
    left,right=x/figw*190,(x+w)/figw*190
    top,bottom=125-(y+h)/figh*125,125-y/figh*125
    bx,by,bw,bh=g['nodes'][node]
    assert bx+.8<=left and right<=bx+bw-.8,(node,left,right)
    assert by+.8<=top and bottom<=by+bh-.8,(node,top,bottom)
feedback=[a for a in g['connectors'] if a['source']=='update' and a['target']=='scale']
assert len(feedback)==1 and feedback[0]['dashed']
x,y,w,h=g['nodes']['scale']
assert feedback[0]['points'][-1]==[x+w/2,y+h]
assert not any(a['source']=='utilisation' for a in g['connectors'])
assert any(a['source']=='exceedance' and a['target']=='update' for a in g['connectors'])
# Open segment samples may not enter any unrelated node interior.
collisions=[]
for ci,c in enumerate(g['connectors']):
    for (x1,y1),(x2,y2) in zip(c['points'],c['points'][1:]):
        n=max(2,int(max(abs(x2-x1),abs(y2-y1))/.1))
        for ni,(bx,by,bw,bh) in g['nodes'].items():
            if ni in [c['source'],c['target']]:continue
            if any(bx+.1<x1+(x2-x1)*j/n<bx+bw-.1 and
                   by+.1<y1+(y2-y1)*j/n<by+bh-.1 for j in range(1,n)):
                collisions.append([ci,ni])
assert not collisions,collisions
ns={'s':'http://www.w3.org/2000/svg'}
svg=ET.parse(OUT/'Fig2.svg')
assert not svg.findall('.//s:image',ns)
assert len(svg.findall('.//s:text',ns))>=40
svg_text=''.join(svg.getroot().itertext())
assert all(c in svg_text for c in ['∈','γ','τ'])
assert any('Cambria' in e.attrib.get('style','') for e in svg.findall('.//s:tspan',ns))
script_spans=[e for e in svg.findall('.//s:tspan',ns)
              if '7.35px' in e.attrib.get('style','') or '7px' in e.attrib.get('style','')]
assert script_spans, 'Expected true reduced-size, baseline-positioned math scripts.'
png=Image.open(OUT/'Fig2.png')
assert png.size==(4488,2952) and abs(png.info['dpi'][0]-600)<.01
sources=json.loads((OUT/'provenance_Fig2.json').read_text(encoding='utf-8'))
checks={'status':'PASS','dimensions_mm':dimensions,'fonts':fonts,
 'minimum_pdf_font_pt':min(s['size'] for s in spans),'raster_images_in_pdf':0,
 'typography':'Cambria math with real scripts; Arial labels',
 'minimum_label_font_pt':8,'math_base_font_pt':'10–10.5 (predictor forms at 8.5)',
 'minimum_math_script_pt':7,'svg_math_script_runs':len(script_spans),
 'png_pixels':list(png.size),'png_dpi':png.info['dpi'],
 'svg_live_text_nodes':len(svg.findall('.//s:text',ns)),
 'node_connector_collisions':collisions,'next_query_feedback_target':'scale bottom anchor',
 'starting_scales':sources['starting_scales']}
(OUT/'verification_Fig2.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
items=['Fig2.pdf','Fig2.svg','Fig2.png','caption_Fig2.md','source/build_fig2.py',
       'source/verify_fig2.py','source/style.py','provenance_Fig2.json']
(OUT/'sha256_Fig2.json').write_text(json.dumps({s:hashlib.file_digest((OUT/s).open('rb'),'sha256').hexdigest()
    for s in items},indent=2),encoding='utf-8')
print('PASS: equations, scales, connectors, Cambria math, real subscripts/superscripts, live SVG text, embedded fonts, 8 pt labels / 7 pt scripts and export size.')
