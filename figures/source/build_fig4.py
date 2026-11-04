"""Applied scale and B-cell request ratio from the stored trajectory, without smoothing."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter
import fitz
from PIL import Image, ImageOps
from style import STYLE

OUT=Path(__file__).resolve().parents[1];PROJECT=OUT.parent
SOURCE=PROJECT/'paper_package_lsad_20261013_r3/data/ad_trajectory.csv'
df=pd.read_csv(SOURCE)
needed=['cell','order','file','direction','temp_C','horizon_s','truth','request','scale','ratio','intervened']
assert all(c in df for c in needed)
assert len(df)==672 and set(df.cell)=={'A1','A17','B'}
frames={cell:df[df.cell.eq(cell)].sort_values('order').reset_index(drop=True) for cell in ['A1','A17','B']}
summary={};segments={}
for cell,expected in [('A1',(368,15,.76,.760,.899)),('A17',(252,10,.82,.782,.932)),('B',(52,3,.90,.790,.880))]:
    d=frames[cell];n,events,start,minimum,last=expected
    assert len(d)==n and d.intervened.sum()==events
    assert d.order.tolist()==list(range(n)) and not d.order.duplicated().any()
    assert np.isfinite(d[['truth','request','scale','ratio']]).all().all() and d.truth.gt(0).all()
    assert abs(d.scale.iloc[0]-start)<1e-12
    assert round(d.scale.min(),3)==minimum and round(d.scale.iloc[-1],3)==last
    assert np.allclose(d.ratio,d.request/d.truth,atol=1e-9,rtol=0)
    assert np.array_equal(d.intervened,d.request>d.truth+1e-9)
    expected_next=np.clip(np.log(d.scale.to_numpy()[:-1])+.05*(.05-d.intervened.to_numpy()[:-1].astype(float)),np.log(.2),np.log(1.5))
    residual=float(np.max(np.abs(np.log(d.scale.to_numpy()[1:])-expected_next)))
    assert residual<2e-10,residual
    group=d[['direction','temp_C','horizon_s']].ne(d[['direction','temp_C','horizon_s']].shift()).any(axis=1).cumsum()
    seg=[]
    for _,block in d.groupby(group,sort=False):
        seg.append({'start':int(block.order.iloc[0]),'end':int(block.order.iloc[-1]),
                    'direction':block.direction.iloc[0],'temp_C':int(block.temp_C.iloc[0]),
                    'duration_s':int(block.horizon_s.iloc[0])})
    segments[cell]=seg
    summary[cell]={'rows':n,'exceedances':events,'start':float(d.scale.iloc[0]),'minimum':float(d.scale.min()),
                   'last_applied':float(d.scale.iloc[-1]),'update_log_residual_max':residual}
b=frames['B']
assert b.loc[b.intervened,'order'].tolist()==[1,2,7]
assert b.iloc[0].request==0 and abs(b.iloc[0].truth-63.803)<1e-9
assert [round(v,3) for v in b.ratio.iloc[1:3]]==[1.258,1.488]
assert [round(b.loc[b.temp_C.eq(t),'ratio'].median(),3) for t in [-20,-10]]==[.737,.799]
assert [(s['start'],s['end'],s['temp_C'],s['duration_s']) for s in segments['B']]==[
 (0,2,25,2),(3,6,25,10),(7,20,-20,10),(21,23,40,10),(24,29,10,10),(30,37,0,10),(38,51,-10,10)]

W,H=190,180
blue=STYLE['online_scale_adaptation'];purple=STYLE['exceedance'];grey=STYLE['reference'];ink=STYLE['text']
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'text.color':ink,'axes.labelcolor':ink,
 'xtick.color':ink,'ytick.color':ink,'axes.edgecolor':STYLE['axes'],'axes.linewidth':.7,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':True,'hatch.linewidth':.6,
 'mathtext.fontset':'custom','mathtext.rm':'Cambria','mathtext.it':'Cambria:italic','mathtext.bf':'Cambria:bold','mathtext.fallback':None})
tstyle=STYLE['temperature_strip'];norm=Normalize(*tstyle['limits_C'])
cmap=LinearSegmentedColormap.from_list('paper_temperature',[tstyle['negative'],tstyle['zero'],tstyle['positive']],N=257)
assert np.allclose(cmap(norm(0))[:3],[1,1,1])
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
axes={};strip_axes={};annotations=[]
def ft(x,y,s,size=8.5,**kwargs):
    return fig.text(x/W,1-y/H,s,fontsize=size,ha='center',va='center',**kwargs)
def panel(label,cell,top,height):
    d=frames[cell]
    ax=fig.add_axes([25/W,(H-top-height)/H,161/W,height/H])
    strip=fig.add_axes([25/W,(H-top-height-3)/H,161/W,3/H],sharex=ax)
    ax.set_xlim(-.6,len(d)-.4)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both',direction='out',labelsize=8,width=.7,length=2.5,pad=2)
    ax.tick_params(axis='x',bottom=False,labelbottom=False)
    strip.set_ylim(0,1);strip.set_yticks([])
    for s in strip.spines.values():s.set_visible(False)
    strip.tick_params(axis='x',direction='out',labelsize=8,width=.7,length=2.5,pad=2)
    ticks=([0,100,200,300,367] if cell=='A1' else [0,50,100,150,200,251] if cell=='A17' else [0,10,20,30,40,51])
    strip.set_xticks(ticks)
    if label=='(d)':
        strip.set_xlabel('Query in processing order (from 0)',fontsize=8.5,labelpad=2)
    for seg in segments[cell]:
        hatch=STYLE['condition_hatches']['short_pulse'] if seg['duration_s']==2 else STYLE['condition_hatches'][seg['direction']]
        strip.add_patch(Rectangle((seg['start']-.5,0),seg['end']-seg['start']+1,1,
                        facecolor=cmap(norm(seg['temp_C'])),edgecolor='#666666',linewidth=.6,hatch=hatch))
    fig.text(4/W,1-(top-3)/H,label,fontsize=10,fontweight='bold',ha='left',va='center')
    fig.text(10.5/W,1-(top-3)/H,f'Cell {cell}',fontsize=8.5,ha='left',va='center')
    axes[label]=ax;strip_axes[label]=strip
    return ax
layout=[('(a)','A1',7,20.5),('(b)','A17',45,20.5),('(c)','B',83,22),('(d)','B',123,22)]
for label,cell,top,height in layout:
    ax=panel(label,cell,top,height);d=frames[cell]
    if label!='(d)':
        ax.set_ylim(.74,.97);ax.set_yticks([.75,.80,.85,.90,.95]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylabel('Scale',fontsize=8.5,labelpad=5)
        ax.axhline(float(d.scale.iloc[0]),color=grey,lw=.8,ls=(0,(3,2)),zorder=1)
        ax.text(len(d)-.9,float(d.scale.iloc[0])+.004,f'Start {float(d.scale.iloc[0]):.2f}',fontsize=8,
                ha='right',va='bottom',color='#666666',zorder=6)
        ax.step(d.order,d.scale,where='post',color=blue,lw=1,zorder=3)
        event=d[d.intervened]
        ax.plot(event.order,event.scale,ls='',marker='v',ms=4.3,mfc=purple,mec=purple,mew=.6,zorder=5)
    else:
        ax.set_ylim(-.08,1.65);ax.set_yticks([0,.4,.8,1.2,1.6]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel('Request /\nreference SOP',fontsize=8.5,labelpad=5)
        ax.axhline(1,color=grey,lw=.8,zorder=1)
        normal=d[~d.intervened];event=d[d.intervened]
        ax.plot(normal.order,normal.ratio,ls='',marker='o',ms=4,mfc=blue,mec=blue,mew=.5,zorder=4)
        ax.plot(event.order,event.ratio,ls='',marker='v',ms=4.3,mfc=purple,mec=purple,mew=.6,zorder=5)
        for order,labeltext,label_x in [(0,'0 W',4),(1,'1.258',5.3),(2,'1.488',7)]:
            xy=(int(order),float(d.iloc[order].ratio))
            pos=(label_x,xy[1])
            ax.annotate(labeltext,xy=xy,xytext=pos,fontsize=8,ha='left',va='center',
                        arrowprops={'arrowstyle':'-','lw':.6,'color':ink,'shrinkA':2,'shrinkB':3})
            annotations.append({'order':order,'label':labeltext,'xy':xy,'xytext':pos})
    if cell=='B':
        for seg in segments['B'][1:]:ax.axvline(seg['start']-.5,color=grey,lw=.6,ls=(0,(1,2)),zorder=0)
        if label=='(c)':
            for seg in segments['B']:
                text=str(seg['temp_C']).replace('-','−')+' °C'
                if seg['duration_s']==2:text+='\n2 s'
                ax.text((seg['start']+seg['end'])/2,1.02,text,transform=ax.get_xaxis_transform(),
                        ha='center',va='bottom',fontsize=8,linespacing=1.1)

cax=fig.add_axes([25/W,12/H,57/W,2.8/H])
cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[-40,-20,0,20,40])
cb.solids.set_rasterized(False)
cb.solids.set_edgecolor('face')
cb.ax.tick_params(labelsize=8,width=.6,length=2,pad=1);cb.outline.set_linewidth(.6)
cb.set_label('Temperature (°C)',fontsize=8.5,labelpad=2)
handles=[Line2D([],[],color=blue,lw=1,marker='o',ms=4,label='Online'),
         Line2D([],[],ls='',marker='v',ms=4.3,color=purple,label='Exceedance'),
         Patch(facecolor='white',edgecolor='#666666',hatch=STYLE['condition_hatches']['charge'],lw=.6,label='Charge'),
         Patch(facecolor='white',edgecolor='#666666',lw=.6,label='Discharge'),
         Patch(facecolor='white',edgecolor='#666666',hatch='xxx',lw=.6,label='2 s')]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(139/W,12/H),ncol=3,frameon=False,
           fontsize=8,handlelength=1.8,handletextpad=.5,columnspacing=1.2,labelspacing=.7)

for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Fig4.{ext}',dpi=600,facecolor='white')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
texts=[{'text':t.get_text(),'size_pt':t.get_fontsize(),'bbox_px':list(t.get_window_extent(renderer).bounds)}
       for t in fig.findobj(match=matplotlib.text.Text) if t.get_visible() and t.get_text()]
geometry={'texts':texts,'figure_pixels':list(fig.bbox.bounds),
 'axes':{k:{'bounds':list(ax.get_position().bounds),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())} for k,ax in axes.items()},
 'annotations':annotations,'temperature_zero_rgb':list(cmap(norm(0))[:3])}
(OUT/'geometry_Fig4.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding='utf-8')
df.sort_values(['cell','order']).to_csv(OUT/'Fig4_plot_data.csv',index=False)
(OUT/'Fig4_condition_segments.json').write_text(json.dumps(segments,indent=2),encoding='utf-8')
sources=[SOURCE,PROJECT/'manuscript.tex']
provenance={'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
 'summary':summary,'B_exceedance_orders':[1,2,7],
 'B_temperature_median_ratios':{str(t):float(b.loc[b.temp_C.eq(t),'ratio'].median()) for t in [-20,-10]},
 'line_interpolation':'post-step; no smoothing','segment_edges':'query index +/-0.5','observations':672}
(OUT/'provenance_Fig4.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
plt.close(fig)
doc=fitz.open(OUT/'Fig4.pdf')
for dpi,name in [(160,'preview'),(96,'print_size_96dpi'),(300,'pdf_render_300dpi')]:
    doc[0].get_pixmap(dpi=dpi,alpha=False).save(OUT/f'Fig4_{name}.png')
im=Image.open(OUT/'Fig4_preview.png').convert('RGB');ImageOps.grayscale(im).save(OUT/'Fig4_grayscale.png')
rgb=np.asarray(im)/255;lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
m=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.01182,.04294,.968881]])
sim=np.clip(lin@m.T,0,1);srgb=np.where(sim<=.0031308,12.92*sim,1.055*sim**(1/2.4)-.055)
Image.fromarray(np.uint8(np.clip(srgb*255,0,255))).save(OUT/'Fig4_deuteranopia.png')
print(json.dumps(provenance['summary'],indent=2))
