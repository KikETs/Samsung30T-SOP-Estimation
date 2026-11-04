"""Draw a reproducible vector schematic. No experimental data are synthesised."""
from pathlib import Path
import json, hashlib, re
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextToPath
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path as MPath
from matplotlib import colors
import fitz
from PIL import Image,ImageOps
from style import STYLE

OUT=Path(__file__).resolve().parents[1]
PROJECT=OUT.parent
W,H=190,125
blue=STYLE['online_scale_adaptation']; grey=STYLE['reference']
purple=STYLE['exceedance']; ink=STYLE['text']
plt.rcParams.update({'font.family':['Arial','Segoe UI','Segoe UI Symbol'],'font.size':8.5,'text.color':ink,
                    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                    'mathtext.fontset':'custom','mathtext.rm':'Cambria',
                    'mathtext.it':'Cambria:italic','mathtext.bf':'Cambria:bold',
                    'mathtext.sf':'Arial','mathtext.cal':'Cambria:italic',
                    'mathtext.fallback':None,
                    'path.simplify':False,'axes.unicode_minus':True})
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
nodes={};connectors=[];text_artists=[]
def text(x,y,s,size=8.5,color=ink,ha='center',weight='normal'):
    t=ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight,
              linespacing=1.3)
    text_artists.append(t);return t
def box(key,x,y,w,h,lines=None,edge=grey,fill='white',size=8.5):
    rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.0',
                       facecolor=fill,edgecolor=edge,lw=.7)
    ax.add_patch(rect);nodes[key]=(x,y,w,h)
    if lines is not None:text(x+w/2,y+h/2,lines,size=size)
    return rect
def anchor(key,side):
    x,y,w,h=nodes[key]
    return {'L':(x,y+h/2),'R':(x+w,y+h/2),'T':(x+w/2,y),'B':(x+w/2,y+h)}[side]
def arrow(points,color=ink,dashed=False,source=None,target=None,width=.8):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7,
                         lw=width,color=color,linestyle=(0,(3,2)) if dashed else '-',
                         capstyle='butt',joinstyle='round')
    ax.add_patch(patch)
    connectors.append({'points':points,'source':source,'target':target,'dashed':dashed})
def link(a,b,sidea='R',sideb='L',**kwargs):
    arrow([anchor(a,sidea),anchor(b,sideb)],source=a,target=b,**kwargs)
def wire(points,color=ink):
    xs,ys=zip(*points);ax.plot(xs,ys,lw=.7,color=color,solid_capstyle='butt')

text(6,6,'(a)',size=10,ha='left',weight='bold')
text(6,51,'(b)',size=10,ha='left',weight='bold')
text(6,90,'(c)',size=10,ha='left',weight='bold')
pale_blue=colors.to_rgba(blue,.07)
pale_purple=colors.to_rgba(purple,.07)
pale_grey=colors.to_rgba(grey,.20)

# (a): P1-P3 follow the resistance path, P4 and P5 the current path, and both meet the power conversion.
box('history',4,21,14,14,r'$I,\,V,\,T$'+'\n'+r'$t < t_k$',edge=blue,fill=pale_blue,size=9.5)
box('features',21,19,21,18,r'$R_{\mathrm{p}},\,R_{\mathrm{r}},\,R_{\mathrm{s}}$'+'\n'+r'$x_j,\,H,\,I_{\mathrm{ref}}$',edge=blue,fill=pale_blue,size=9.5)
box('region',45,21,10,14,edge=blue,fill=pale_blue)
text(50,25,r'$V_{\mathrm{pre}}$',size=9.5)
text(50,31,'Region',size=8)
link('history','features',color=blue);link('features','region',color=blue)
ax.add_patch(FancyBboxPatch((61.5,7.5),25,33,boxstyle='round,pad=0,rounding_size=1',
                           facecolor=pale_grey,edgecolor=grey,lw=.7))
labels=[r'Power-law $\hat{R}$',r'$\log\hat{R}$',r'$\log\hat{R}$',r'$\log\hat{I}$',r'$\log\hat{I}$, $H-\delta$']
ys=[11,17,23,29,35]
for i,(y,label) in enumerate(zip(ys,labels),1):
    nodes[f'P{i}']=(61.5,y-2.5,25,5)
    text(63.5,y,f'P{i}',size=8.5,ha='left')
    text(69,y,label,size=8.5,ha='left')
    if i<5:wire([(63.5,y+3),(84.5,y+3)],ink if i==3 else grey)
wire([(55,28),(57.5,28)],blue);wire([(57.5,11),(57.5,35)],blue)
for i,y in enumerate(ys,1):
    arrow([(57.5,y),anchor(f'P{i}','L')],color=blue,target=f'P{i}')
box('respath',90.5,8.5,28,17,edge=blue,fill=pale_blue)
text(104.5,12.8,r"$\hat{I}\,\hat{R}(\hat{I})=H'$",size=9.5)
text(104.5,18.4,'P1: iteration',size=8)
text(104.5,22.2,'P2, P3: closed form',size=8)
box('curpath',90.5,27.5,28,11,edge=blue,fill=pale_blue)
text(104.5,31,r'$\hat{I}$ from the',size=8.5)
text(104.5,35,'regression',size=8.5)
for i,y in enumerate(ys,1):
    arrow([anchor(f'P{i}','R')[0:1]+(y,),(90.5,y)],color=blue,source=f'P{i}',target='respath' if i<=3 else 'curpath')
box('power',122.5,8.5,31,26,edge=blue,fill=pale_blue)
text(138,12.6,r"$\hat{p}_i=V'\hat{I}$",size=9.5)
text(138,17.6,r'or $I_{\max}(V_{\mathrm{pre}}\pm I_{\max}\hat{R})$',size=8.5)
text(138,21.8,r'when $\hat{I}>I_{\max}$',size=8)
text(138,26.6,r'$\hat{R}=\hat{R}(I_{\max})$: P1–P3',size=8)
text(138,30.6,r"$\hat{R}=H'/\hat{I}$: P4, P5",size=8)
arrow([anchor('respath','R'),(122.5,17)],color=blue,source='respath',target='power')
arrow([(118.5,33),(122.5,33)],color=blue,source='curpath',target='power')
box('combined',122.5,39,40,8,edge=blue,fill=pale_blue)
text(142.5,43,r'$\hat{p}_k=\exp[\mathrm{median}_i(\log\hat{p}_i)]$',size=9.5)
text(142.5,50.2,r'over finite $\hat{p}_i>0$ (none: $p_k=0$ W)',size=8)
arrow([anchor('power','B'),(138,39)],color=blue,source='power',target='combined')
nodes['scale']=(167.5,38,10,10)
ax.add_patch(Circle((172.5,43),5,facecolor=pale_blue,edgecolor=blue,lw=.8))
text(172.5,43,r'$\times\,s_k$',size=9.5,color=blue)
link('combined','scale',color=blue)
nodes['request']=(181.5,39,7,8)
text(185,43,r'$p_k$',size=10.5,color=blue)
link('scale','request',color=blue)

# (b): only exceedance controls the next scale; utilisation is a terminal metric.
box('inputs',6,59,19,22,fill=pale_grey)
text(15.5,65,r'$p_k$',color=blue,size=10.5)
text(15.5,75,r'$y_k$',size=10.5)
box('exceedance',36,55,58,11,r'$e_k=\mathbf{1}[p_k/y_k>1+10^{-9}]$',
    edge=purple,fill=pale_purple,size=10.5)
box('utilisation',36,74,58,10,r'$u_k=\min[p_k/y_k,\,1]$',size=10.5)
wire([anchor('inputs','R'),(30,70)])
wire([(30,60.5),(30,79)])
arrow([(30,60.5),anchor('exceedance','L')],color=purple,target='exceedance')
arrow([(30,79),anchor('utilisation','L')],target='utilisation')
box('update',103,55,58,26,edge=blue,fill=pale_blue)
text(107,60,r'$\log s_{k+1}=\mathrm{clip}[\log s_k+\gamma(\tau^{*}-e_k),$',size=10.5,ha='left')
text(130,66.5,r'$\log 0.20,\,\log 1.50]$',size=10.5,ha='left')
text(132,76,r'$\gamma=0.05,\quad\tau^{*}=5\,\%$',size=10.5)
arrow([anchor('exceedance','R'),(98.5,60.5),(98.5,68),anchor('update','L')],
      color=purple,source='exceedance',target='update')
arrow([anchor('update','R'),(172.5,68),anchor('scale','B')],
      color=blue,dashed=True,source='update',target='scale')
text(182,57,r'$k\to k+1$',size=10,color=blue)
text(182,62,'Next query',size=8,color=blue)

# (c): calibration follows the same left-to-right reading direction as prediction.
for x,label,fill,edge in [(6,'Evaluation','white',blue),(29,'Training',pale_grey,grey)]:
    ax.add_patch(Rectangle((x,94),3,3,facecolor=fill,edgecolor=edge,lw=.7))
    text(x+4,95.5,label,size=8,ha='left')
cells=['A1','A17','B'];tilex=[6,19,32]
trajpath=PROJECT/'paper_package_lsad_20261013_r3/data/ad_trajectory.csv'
trajectory=pd.read_csv(trajpath)
scales=trajectory.sort_values(['cell','order']).groupby('cell').first()['scale'].to_dict()
assert all(abs(scales[c]-v)<1e-12 for c,v in zip(cells,[.76,.82,.90]))
box('folds',4,98,41,24,edge=grey)
for row,(ev,y) in enumerate(zip(cells,[99,107,115])):
    for cell,x in zip(cells,tilex):
        box(f'fold{row}_{cell}',x,y,11,6,cell,edge=blue if cell==ev else grey,
            fill='white' if cell==ev else colors.to_rgba(grey,.35),size=8)
box('fit',52,99,57,22,fill=pale_grey)
text(80.5,95.5,r'$C_1,\,C_2$',size=10.5)
text(80.5,103,r'P1–P5, $V_{\mathrm{pre}}$ boundaries',size=10)
text(80.5,110,r'Discharge: $C_1\to C_2,\ C_2\to C_1$',size=10)
text(80.5,117,r'Charge: $\{C_1,C_2\}\to\{C_1,C_2\}$',size=10)
link('folds','fit')
box('calibration',116,99,52,22,edge=blue,fill=pale_blue)
text(142,102.4,r'$s_0=\max\{s\in S:E_{\mathrm{cal}}(s)\leq\tau\}$',size=10.5)
text(142,107.6,r'($s_0=0.20$ if no $s$ qualifies)',size=8.5)
text(142,112.9,r'$S=\{0.20,\,0.21,\,\ldots,\,1.50\}$',size=10.5)
text(142,118.2,r'$\tau=5\,\%$',size=10.5)
link('fit','calibration',color=blue)
text(181.5,95.5,r'$s_0$',size=10.5,color=blue)
nodes['initial_scales']=(174,99,15,22)
for ev,y in zip(cells,[102,110,118]):
    text(181.5,y,f'{ev}  {scales[ev]:.2f}',size=8,color=blue)
link('calibration','initial_scales',color=blue)

# Export consistent physical bounds without tight-bbox resizing.
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'Fig2.{ext}',dpi=600,facecolor='white')
# Some SVG consumers do not perform per-character fallback from CSS font lists.
# Explicit live-text runs preserve the exact fonts used by the PDF renderer.
ns='http://www.w3.org/2000/svg'
ET.register_namespace('',ns);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
tree=ET.parse(OUT/'Fig2.svg')
families=['Arial','Segoe UI','Segoe UI Symbol']
cmaps={f:TTFont(font_manager.findfont(f)).getBestCmap() for f in families}
text_metrics=TextToPath()
for element in tree.findall(f'.//{{{ns}}}text'):
    value=element.text or ''
    # Mathtext already emits positioned font runs and real baseline shifts.
    if len(element) or not value.strip():continue
    if all(ord(ch) in cmaps['Arial'] for ch in value):continue
    runs=[]
    for ch in value:
        family=next((f for f in families if ord(ch) in cmaps[f]),None)
        assert family is not None,repr(ch)
        if runs and runs[-1][0]==family:runs[-1][1]+=ch
        else:runs.append([family,ch])
    style=element.attrib['style']
    size=float(re.search(r'font:\s*([\d.]+)px',style).group(1))
    widths=[text_metrics.get_text_width_height_descent(part,
        font_manager.FontProperties(family=family,size=size),ismath=False)[0]
        for family,part in runs]
    cursor=float(element.attrib.get('x','0'))
    if 'text-anchor: middle' in style:cursor-=sum(widths)/2
    elif 'text-anchor: end' in style:cursor-=sum(widths)
    element.attrib['style']=re.sub(r'text-anchor: (middle|end)','text-anchor: start',style)
    element.text=None
    for (family,part),width in zip(runs,widths):
        span=ET.SubElement(element,f'{{{ns}}}tspan',{
            'x':str(cursor),'style':f"font-family: '{family}'; text-anchor: start"})
        span.text=part
        cursor+=width
tree.write(OUT/'Fig2.svg',encoding='utf-8',xml_declaration=True)
fig.canvas.draw();r=fig.canvas.get_renderer()
text_bounds=[{'text':t.get_text(),'size_pt':t.get_fontsize(),
              'bounds_pixels':list(t.get_window_extent(r).bounds)} for t in text_artists]
(OUT/'geometry_Fig2.json').write_text(json.dumps({'nodes':nodes,'connectors':connectors,
    'texts':text_bounds,'figure_pixels':list(fig.bbox.bounds)},indent=2,ensure_ascii=False),encoding='utf-8')
sources=[PROJECT/'manuscript.tex',
         PROJECT/'tables/T2_predictors.tex',trajpath]
(OUT/'provenance_Fig2.json').write_text(json.dumps({'sources':[{'path':str(p),
    'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in sources],
    'starting_scales':scales,'schematic':True,'data_plots':False,
    },indent=2),encoding='utf-8')
plt.close(fig)
doc=fitz.open(OUT/'Fig2.pdf');page=doc[0]
for dpi,name in [(160,'preview'),(96,'print_size_96dpi'),(300,'pdf_render_300dpi')]:
    page.get_pixmap(dpi=dpi,alpha=False).save(OUT/f'Fig2_{name}.png')
im=Image.open(OUT/'Fig2_preview.png').convert('RGB');ImageOps.grayscale(im).save(OUT/'Fig2_grayscale.png')
rgb=np.asarray(im)/255;lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
matrix=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.01182,.04294,.968881]])
sim=np.clip(lin@matrix.T,0,1);srgb=np.where(sim<=.0031308,12.92*sim,1.055*sim**(1/2.4)-.055)
Image.fromarray(np.uint8(np.clip(srgb*255,0,255))).save(OUT/'Fig2_deuteranopia.png')
print(json.dumps({'size_mm':[page.rect.width*25.4/72,page.rect.height*25.4/72],
 'fonts':page.get_fonts(),'images':len(page.get_images()),'png':Image.open(OUT/'Fig2.png').size},indent=2))
