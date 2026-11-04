"""Rebuild Figure 1 from the archived original CSV, with explicit data QC.

Run fetch_raw.py once, then run this script. No measurement is interpolated,
smoothed, resampled or synthesised. Measured points are connected by straight
segments. Panel (c) is an explicitly non-quantitative information schematic.
"""
from pathlib import Path
import json, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import fitz
from PIL import Image, ImageOps
from style import STYLE as S

OUT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_data.py'))],check=True,stdout=subprocess.DEVNULL)
meta=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
w=pd.read_csv(OUT/'data/query_window.csv')
trials=pd.read_csv(OUT/'data/search_pulses.csv')
plt.rcParams.update({
    'font.family':S['font'],'font.size':8,'axes.labelsize':8.5,
    'axes.linewidth':.65,'axes.edgecolor':S['axes'],
    'text.color':S['text'],'axes.labelcolor':S['text'],
    'xtick.color':S['text'],'ytick.color':S['text'],
    'xtick.labelsize':8,'ytick.labelsize':8,
    'xtick.direction':'out','ytick.direction':'out',
    'xtick.major.size':2.5,'ytick.major.size':2.5,
    'xtick.major.width':.65,'ytick.major.width':.65,
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'mathtext.fontset':'custom','mathtext.rm':S['font'],
    'mathtext.it':S['font']+':italic','mathtext.bf':S['font']+':bold',
    'axes.unicode_minus':True,'path.simplify':False,
})
blue=S['online_scale_adaptation']; grey=S['reference']; ink=S['text']
fig=plt.figure(figsize=(S['width_mm']/25.4,S['height_mm']/25.4),facecolor='white')
av=fig.add_axes([.085,.730,.885,.175])
ap=fig.add_axes([.085,.535,.885,.150],sharex=av)
b=fig.add_axes([.085,.095,.292,.298])
c=fig.add_axes([.535,.075,.435,.335]);c.set_axis_off();c.set_xlim(0,1);c.set_ylim(-.07,1.04)
for ax in [av,ap,b]:
    ax.spines[['right','top']].set_visible(False)
    ax.tick_params(pad=3)

# Panel labels: identical offsets and baselines within each layout row.
fig.text(.025,.947,'(a)',fontsize=10,fontweight='bold',va='top')
fig.text(.025,.433,'(b)',fontsize=10,fontweight='bold',va='top')
fig.text(.475,.433,'(c)',fontsize=10,fontweight='bold',va='top')

legend=[Patch(facecolor=blue,alpha=.10,edgecolor=blue,label='Pre-query measurements'),
        Patch(facecolor=grey,alpha=.30,edgecolor=grey,label='Final reference pulse'),
        Line2D([],[],color=grey,lw=.75,ls='--',label='2.55 V ± 40 mV')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.54,.972),ncol=3,
           frameon=False,handlelength=1.6,columnspacing=1.7,fontsize=8)
x=w.query_time_s.to_numpy()
for ax,col in [(av,'Voltage(V)'),(ap,'power_magnitude_W')]:
    ax.axvspan(-900,0,facecolor=blue,alpha=.10,zorder=0)
    ax.axvspan(0,10,facecolor=grey,alpha=.30,zorder=0)
    ax.plot(x,w[col],color=ink,lw=.70,zorder=2)
    ax.set_xlim(-900,42)
av.axhspan(2.51,2.59,color=grey,alpha=.30,zorder=0)
av.axhline(2.55,color='#777777',lw=.65,ls='--',zorder=1)
av.set_ylim(2.45,3.70);av.set_yticks([2.5,3.0,3.5]);av.set_ylabel('Voltage (V)')
av.tick_params(axis='x',bottom=False,labelbottom=False)
ap.set_ylim(-4,106);ap.set_yticks([0,50,100]);ap.set_ylabel('|V·I| (W)')
ap.set_xticks([-900,-600,-300,0]);ap.set_xlabel('Time relative to final pulse start (s)',labelpad=4)

# Symbols and short callouts identify measured events; long explanations are in caption.
prevx=meta['preceding_pulse']['end_absolute_time_s']-meta['query']['start_absolute_time_s']
ap.plot([prevx],[meta['preceding_pulse']['end_power_W']],marker='o',ms=3.8,
        markerfacecolor='white',markeredgecolor=ink,markeredgewidth=.7,zorder=4)
ap.annotate('Preceding pulse',xy=(prevx,84.2014),xytext=(-600,92),fontsize=8,
            arrowprops=dict(arrowstyle='-',lw=.65,color=ink),ha='center',va='bottom')
av.plot([-298,-2],[3.57,3.57],color=ink,lw=.65)
av.plot([-298,-298],[3.54,3.57],color=ink,lw=.65)
av.plot([-2,-2],[3.54,3.57],color=ink,lw=.65)
av.text(-150,3.59,'Rest 299.019 s',ha='center',va='bottom',fontsize=8)
av.plot(10,meta['query']['end_voltage_V'],marker='D',ms=4,color='black',zorder=5)
ap.plot(10,meta['query']['end_power_VI_W'],marker='D',ms=4,color='black',zorder=5)
ap.annotate('Reference SOP 65.6 W',xy=(10,65.6),xytext=(-155,96),ha='center',va='center',
            arrowprops=dict(arrowstyle='-',lw=.65,color=ink),fontsize=8)
request=float(meta['request']['request'])
ap.plot([14,36],[request,request],color=blue,lw=1.1,ls='--',zorder=5)
ap.annotate('Request 57.1 W',xy=(28,request),xytext=(-135,32),ha='center',va='center',
            color=blue,fontsize=8)

# Search trials were applied in successive full sweeps at the same SOC.
b.axhspan(2.51,2.59,color=grey,alpha=.30,zorder=0)
b.axhline(2.55,color='#777777',lw=.65,ls='--',zorder=1)
linestyles=[(0,(1,1.8)), '-', (0,(5,3)), '-', (0,(3,2.4))]
trial_handles=[]
for j,(trial,q) in enumerate(trials.groupby('trial',sort=True)):
    last=trial==5
    powr=meta['trials'][trial-1]['power_label_W']
    line,=b.plot(q.pulse_time_s,q['Voltage(V)'],color=S['search_trial_colors'][trial],
           lw=1.35 if last else 1.15,ls=linestyles[j],marker='D' if last else None,
           markevery=[len(q)-1] if last else None,ms=4,
           markerfacecolor='black' if last else 'white',
           markeredgecolor='black' if last else '#777777',markeredgewidth=.65,
           zorder=5 if last else 2+j*.1,clip_on=False,label=f'{trial}: {powr:.1f} W')
    trial_handles.append(line)
# Match each actual line encoding to its chronological trial and power.
b.legend(handles=trial_handles,title='Iteration / power',title_fontsize=8,
         loc='upper left',bbox_to_anchor=(1.035,1.0),borderaxespad=0,
         frameon=False,fontsize=8,handlelength=2.6,handletextpad=.6,
         labelspacing=.75,numpoints=1)
b.set_xlim(0,10);b.set_xticks([0,2,4,6,8,10]);b.set_ylim(2.30,3.30)
b.set_yticks([2.4,2.6,2.8,3.0,3.2]);b.set_ylabel('Voltage (V)')
b.set_xlabel('Time since pulse start (s)',labelpad=4)

# Resolve near-coincident pairs using differences at the two exact shared
# recorded times. No time alignment, interpolation or voltage offset is used.
detail=fig.add_axes([.145,.155,.085,.075])
detail.spines[['right','top']].set_visible(False)
detail.tick_params(labelsize=8,pad=2,length=2)
diff_rows=[]
for left,right,color,ls in [(2,3,ink,(0,(5,3))),(4,5,blue,(0,(3,2.4)))]:
    qa=trials[(trials.trial==left)&trials.pulse_time_s.isin([9.0,10.0])]
    qb=trials[(trials.trial==right)&trials.pulse_time_s.isin([9.0,10.0])]
    pair=qa[['pulse_time_s','Voltage(V)']].merge(
        qb[['pulse_time_s','Voltage(V)']],on='pulse_time_s',suffixes=('_left','_right'))
    assert pair.pulse_time_s.tolist()==[9.0,10.0]
    delta=1000*(pair['Voltage(V)_left']-pair['Voltage(V)_right'])
    detail.plot(pair.pulse_time_s,delta,color=color,ls=ls,lw=1.1)
    detail.text(10.12,delta.iloc[-1],f'{left}−{right}',fontsize=8,color=color,
                va='center',clip_on=False)
    diff_rows.extend([{'left_trial':left,'right_trial':right,'time_s':float(t),
                      'voltage_difference_mV':float(v)} for t,v in zip(pair.pulse_time_s,delta)])
detail.set_xlim(8.95,10.05);detail.set_ylim(0,5)
detail.set_xticks([9,10]);detail.set_yticks([0,4])
detail.set_xlabel('Time (s)',fontsize=8,labelpad=2)
detail.set_ylabel('ΔV (mV)',fontsize=8,labelpad=2)
pd.DataFrame(diff_rows).to_csv(OUT/'data/endpoint_differences.csv',index=False)

def arr(src,dst,color=ink,ls='-',lw=.75,scale=8):
    patch=FancyArrowPatch(src,dst,arrowstyle='-|>',mutation_scale=scale,
                          lw=lw,color=color,linestyle=ls,shrinkA=1,shrinkB=1,
                          transform=c.transData,clip_on=False)
    c.add_patch(patch)
    return patch
def node(x,y,text,width,height=.12,color=ink,face='white'):
    c.add_patch(Rectangle((x-width/2,y-height/2),width,height,
                         facecolor=face,edgecolor=color,lw=.65))
    c.text(x,y,text,ha='center',va='center',fontsize=8,color=color)

# Schematic timeline; icons have no quantitative amplitude or temporal scale.
positions=[.075,.30,.58,.88]
c.plot([.025,.47],[.985,.985],color=blue,lw=.8)
c.plot([.025,.025],[.96,.985],color=blue,lw=.8)
c.plot([.47,.47],[.96,.985],color=blue,lw=.8)
c.text(.245,.915,r"$V,\ T,\ V_{\mathrm{relax}},\ \{V,I\}_{k'<k}$",
       color=blue,ha='center',va='center',fontsize=8)
arr((.005,.76),(.99,.76),scale=7)
for j,pos in enumerate(positions):
    col=blue if j<2 else grey
    c.plot([pos-.03,pos-.03,pos+.03,pos+.03],[.76,.835,.835,.76],
           lw=1,color=col,ls='-' if j<3 else '--')
    c.text(pos,.685,[r'$k-2$',r'$k-1$',r'$k$',r'$k+1$'][j],ha='center',va='center')
c.plot([.50,.50],[.64,1.01],color=grey,lw=.7,ls=':')

node(.16,.47,r'$e_{k-2},\ e_{k-1}$',.29)
node(.44,.47,r'$s_k$',.13,color=blue)
node(.79,.47,r'$p_k=s_k\hat{p}_k$',.34,color=blue)
arr((.305,.47),(.375,.47),color=blue)
arr((.505,.47),(.62,.47),color=blue)
node(.35,.23,r'$y_k$',.14,color='#777777')
node(.77,.23,r'$e_k=\mathbf{1}[p_k/y_k>1+10^{-9}]$',.44)
arr((.42,.23),(.55,.23))
arr((.79,.41),(.79,.29),color=blue)
node(.79,0,r'$s_{k+1}$',.19,height=.10,color=blue)
arr((.79,.17),(.79,.05),ls=(0,(1.6,1.4)),color=blue)
arr((.11,-.02),(.23,-.02),ls=(0,(1.6,1.4)),color=blue)
c.text(.30,-.02,'Feedback',ha='left',va='center',fontsize=8)

# Inspect the same fixed-size canvas in every output. No tight-bbox size changes.
for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'Fig1.{ext}',dpi=600,facecolor='white')
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
text_bounds=[]
for ax in fig.axes:
    for txt in ax.texts:
        bb=txt.get_window_extent(renderer)
        text_bounds.append({'text':txt.get_text(),'bbox_px':list(bb.bounds)})
(OUT/'text_geometry.json').write_text(json.dumps(text_bounds,indent=2),encoding='utf-8')
plt.close(fig)
doc=fitz.open(OUT/'Fig1.pdf');page=doc[0]
for dpi,name in [(160,'Fig1_preview.png'),(96,'Fig1_print_size_96dpi.png'),(300,'Fig1_pdf_render_300dpi.png')]:
    page.get_pixmap(dpi=dpi,alpha=False).save(OUT/name)
im=Image.open(OUT/'Fig1_preview.png').convert('RGB')
ImageOps.grayscale(im).save(OUT/'Fig1_grayscale.png')
# Machado et al. full-severity deuteranopia approximation in linear RGB.
rgb=np.asarray(im)/255
lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
matrix=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.01182,.04294,.968881]])
sim=np.clip(lin@matrix.T,0,1)
srgb=np.where(sim<=.0031308,12.92*sim,1.055*sim**(1/2.4)-.055)
Image.fromarray(np.uint8(np.clip(srgb*255,0,255))).save(OUT/'Fig1_deuteranopia.png')
print(json.dumps({'pdf_mm':[page.rect.width*25.4/72,page.rect.height*25.4/72],
 'png_size':Image.open(OUT/'Fig1.png').size,'fonts':page.get_fonts(),
 'pdf_images':len(page.get_images()),'samples_panel_a':len(w)},indent=2))
