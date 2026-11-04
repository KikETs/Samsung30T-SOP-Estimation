"""Fig. 5: local-predictor curves over the calibration-target sweep, with the comparators."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz
from PIL import Image,ImageOps
from style import STYLE

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
paths={
 'local':ROOT/'paper_package_lsad_20261013_r3/data/iso_tradeoff_posthoc.csv',
 'pulse':ROOT/'comparisons/20261014_baseline_pilot/results/curves.csv',
 'circuit':ROOT/'comparisons/20261104_ecm_rls/results/curves.csv',
 'table':ROOT/'tables/csv/T5_comparators.csv',
 'manuscript':ROOT/'manuscript.tex'}
local=pd.read_csv(paths['local']);table=pd.read_csv(paths['table'])
local.to_csv(OUT/'Fig5_local_plot_data.csv',index=False)
cols=['tau','exceed_rows','int_rate','util_q50','closed_loop']
FIXED='Fixed';ONLINE='Online';LOC='Five local predictors';CHEN='Reproduction of Chen et al.'
parts=[]
for raw,policy in [('fixed scale',FIXED),('online adaptation (tau* = tau)',ONLINE)]:
    d=local[local.policy.eq(raw)][cols].copy();d['method']=LOC;d['scale']=policy;parts.append(d)
maps=[('pulse','method',{'PERSISTENCE':'Previous pulse power','RINT_PREVIOUS':'Previous pulse resistance'}),
      ('circuit','candidate',{'ECM-1RC-RLS-A1':'Equivalent circuit, one RC branch','ECM-2RC-RLS-A1':'Equivalent circuit, two RC branches'})]
for source,key,names in maps:
    raw=pd.read_csv(paths[source])
    for code,name in names.items():
        for pol,scale in [('fixed',FIXED),('AD',ONLINE)]:
            d=raw[raw[key].eq(code)&raw.policy.eq(pol)][cols].copy()
            d['method']=name;d['scale']=scale;parts.append(d)
curves=pd.concat(parts,ignore_index=True)
for _,d in curves.groupby(['method','scale']):
    assert len(d)==81 and np.allclose(d.sort_values('tau').tau,np.arange(81)/400,atol=1e-14,rtol=0)
    assert np.isfinite(d[cols]).all().all()
    assert np.allclose(d.int_rate,d.exceed_rows/672,atol=1e-14,rtol=0)
chen=table[table.predictor.str.startswith(CHEN)].iloc[0]
point=pd.DataFrame([{'tau':.05,'exceed_rows':int(chen.fixed_exceedances),
 'int_rate':chen.fixed_exceedances/672,'util_q50':chen.fixed_median_utilisation_pct/100,
 'closed_loop':int(chen.fixed_util_exc_groups),'method':CHEN,'scale':FIXED}])
plot=pd.concat([curves,point],ignore_index=True);plot.to_csv(OUT/'Fig5_plot_data.csv',index=False)
local_curves=curves[curves.method.eq(LOC)];fixed=local_curves[local_curves.scale.eq(FIXED)]
front=fixed.groupby('exceed_rows')[['util_q50','closed_loop']].max().cummax().reset_index()
env=local_curves.groupby('exceed_rows')[['util_q50','closed_loop']].max().cummax().reset_index()
front.to_csv(OUT/'Fig5_fixed_frontier.csv',index=False);env.to_csv(OUT/'Fig5_local_envelope.csv',index=False)
comp=curves[~curves.method.eq(LOC)].copy()
for col in ['util_q50','closed_loop']:
    comp['local_envelope_'+col]=[local_curves.loc[local_curves.exceed_rows.le(n),col].max() for n in comp.exceed_rows]
    comp['above_'+col]=comp[col]>comp['local_envelope_'+col]+1e-12
    assert not comp['above_'+col].any()
comp.to_csv(OUT/'Fig5_envelope_comparison.csv',index=False)
for (method,scale),d in curves.groupby(['method','scale']):
    t=table[table.predictor.eq(method)].iloc[0];m=d[np.isclose(d.tau,.05)].iloc[0]
    pref='fixed' if scale==FIXED else 'adapted'
    assert int(m.exceed_rows)==int(t[pref+'_exceedances'])
    assert abs(m.util_q50*100-t[pref+'_median_utilisation_pct'])<.000051
    assert int(m.closed_loop)==int(t[pref+'_util_exc_groups'])

W,H=190,170
plt.rcParams.update({'font.family':STYLE['font'],'font.size':9.3,'text.color':STYLE['text'],
 'axes.labelcolor':STYLE['text'],'xtick.color':STYLE['text'],'ytick.color':STYLE['text'],
 'axes.edgecolor':STYLE['axes'],'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,
 'axes.unicode_minus':True,'mathtext.fontset':'custom','mathtext.rm':'Cambria',
 'mathtext.it':'Cambria:italic','mathtext.bf':'Cambria:bold','mathtext.fallback':None})
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white');axes={};annotations=[]
def make_axes(label,left,top,col):
    ax=fig.add_axes([left/W,(H-top-44)/H,73/W,44/H]);axes[label]=ax
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.set(xlim=(-.3,21.3),xticks=[0,5,10,15,20],
           ylim=(35,96) if col=='util_q50' else (-.45,15),
           yticks=[40,50,60,70,80,90] if col=='util_q50' else [0,3,6,9,12,15])
    ax.set_xlabel('Realised exceedance rate (%)',fontsize=9.3,labelpad=4)
    ax.set_ylabel('Median utilisation (%)' if col=='util_q50' else 'Passing groups',fontsize=9.3,labelpad=5)
    ax.tick_params(direction='out',labelsize=9.3,width=.7,length=2.5,pad=2)
    ax.axvline(28/6.72,color=STYLE['reference'],lw=.8,ls=(0,(1,2)),zorder=0)
    fig.text((left-10)/W,1-(top-4)/H,label,fontsize=10,fontweight='bold',ha='left',va='center')
    return ax
def step(ax,data,col,mult,color,ls,lw):
    x=np.r_[data.exceed_rows.to_numpy()/6.72,21.3]
    y=np.r_[data[col].to_numpy()*mult,data[col].iloc[-1]*mult]
    ax.step(x,y,where='post',color=color,ls=ls,lw=lw,zorder=1)
colors={FIXED:STYLE['fixed_scale'],ONLINE:STYLE['online_scale_adaptation']}
for label,left,col,mult in [('(a)',18,'util_q50',100),('(b)',112,'closed_loop',1)]:
    ax=make_axes(label,left,9,col);step(ax,front,col,mult,colors[FIXED],(0,(4,2)),.9)
    for policy in [FIXED,ONLINE]:
        d=local_curves[local_curves.scale.eq(policy)].sort_values('tau');r=d[np.isclose(d.tau,.05)].iloc[0]
        marker='s' if policy==FIXED else 'o';face='white' if policy==FIXED else colors[policy]
        ax.plot(d.int_rate*100,d[col]*mult,color=colors[policy],lw=.6,marker=marker,ms=4,mfc=face,mec=colors[policy],mew=.8,zorder=3)
        xy=(float(r.int_rate*100),float(r[col]*mult))
        ax.plot(*xy,ls='',marker=marker,ms=6,mfc=face,mec=STYLE['text'],mew=.9,zorder=5)
        pos={'(a)':{FIXED:(4,69),ONLINE:(3.4,94)},'(b)':{FIXED:(4,5.8),ONLINE:(6.3,13.5)}}[label][policy]
        ha='right' if policy==ONLINE and label=='(a)' else 'left'
        connection='angle,angleA=180,angleB=90,rad=0' if policy==ONLINE and label=='(b)' else 'arc3,rad=0'
        ax.annotate(r'$\tau=5\%$',xy=xy,xytext=pos,fontsize=10,ha=ha,va='center',
                    arrowprops={'arrowstyle':'-','lw':.6,'color':STYLE['text'],'shrinkA':3,'shrinkB':4,
                                'relpos':(1,.5) if ha=='right' else (0,.5),'connectionstyle':connection},zorder=6)
        annotations.append({'panel':label,'method':LOC,'scale':policy,'xy':xy,'xytext':pos})
for label,left,col,mult in [('(c)',18,'util_q50',100),('(d)',112,'closed_loop',1)]:
    ax=make_axes(label,left,77,col);step(ax,env,col,mult,STYLE['reference'],'-',1.2)
    for name,sty in STYLE['comparators'].items():
        for policy in [FIXED,ONLINE]:
            d=curves[curves.method.eq(name)&curves.scale.eq(policy)].sort_values('tau')
            face='white' if policy==FIXED else sty['color'];line=(0,(4,2)) if policy==FIXED else '-'
            ax.plot(d.int_rate*100,d[col]*mult,color=sty['color'],lw=.65,ls=line,
                    marker=sty['marker'],ms=3,mfc=face,mec=sty['color'],mew=.6,zorder=3)
    # Main markers are layered after every curve, preserving identical observed positions.
    for name,sty in STYLE['comparators'].items():
        for policy in [FIXED,ONLINE]:
            r=curves[curves.method.eq(name)&curves.scale.eq(policy)&np.isclose(curves.tau,.05)].iloc[0]
            ax.plot(r.int_rate*100,r[col]*mult,ls='',marker=sty['marker'],ms=5.2,
                    mfc='white' if policy==FIXED else sty['color'],mec=STYLE['text'],mew=.8,zorder=5)
    ax.plot(point.int_rate*100,point[col]*mult,ls='',marker='*',ms=7.5,color=STYLE['text'],mew=.7,zorder=6)
top_handles=[Line2D([],[],color=colors[FIXED],lw=.6,marker='s',ms=4,mfc='white',label='Fixed'),
 Line2D([],[],color=colors[ONLINE],lw=.6,marker='o',ms=4,label='Online'),
 Line2D([],[],color=colors[FIXED],lw=.9,ls=(0,(4,2)),label='Fixed frontier')]
fig.legend(handles=top_handles,loc='center',bbox_to_anchor=(.5,1-65/H),ncol=3,frameon=False,fontsize=9.3,
           handlelength=2.6,handletextpad=.6,columnspacing=2)
# Separate named-method key from the shared scale encoding to avoid repeating ten long names.
method_handles=[Line2D([],[],ls='',marker=s['marker'],ms=4,color=s['color'],label=n) for n,s in STYLE['comparators'].items()]
method_handles += [Line2D([],[],ls='',marker='*',ms=7,color=STYLE['text'],label=CHEN),
                   Line2D([],[],color=STYLE['reference'],lw=1.2,label='Local predictors, best at ≤ rate')]
# Matplotlib fills columns first; explicit reordering creates three balanced reading rows.
method_handles=[method_handles[i] for i in [0,2,4,1,3,5]]
fig.legend(handles=method_handles,loc='center',bbox_to_anchor=(.5,1-146/H),ncol=2,frameon=False,fontsize=9.3,
           handlelength=2.1,handletextpad=.6,columnspacing=1.8,labelspacing=.75)
scale_handles=[Line2D([],[],color=STYLE['text'],lw=.65,ls=(0,(4,2)),marker='o',ms=3,mfc='white',label='Fixed'),
 Line2D([],[],color=STYLE['text'],lw=.65,marker='o',ms=3,label='Online'),
 Line2D([],[],ls='',color=STYLE['text'],marker='o',ms=5.2,mfc=STYLE['reference'],mec=STYLE['text'],label=r'$\tau=5\%$')]
fig.legend(handles=scale_handles,loc='center',bbox_to_anchor=(.5,1-162/H),ncol=3,frameon=False,fontsize=9.3,
           handlelength=2.6,handletextpad=.6,columnspacing=2)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Fig5.{ext}',dpi=600,facecolor='white')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
geometry={'figure_pixels':list(fig.bbox.bounds),'annotations':annotations,
 'axes':{k:{'bounds':list(ax.get_position().bounds),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())} for k,ax in axes.items()},
 'texts':[{'text':t.get_text(),'size_pt':t.get_fontsize(),'bbox_px':list(t.get_window_extent(renderer).bounds)}
          for t in fig.findobj(match=matplotlib.text.Text) if t.get_visible() and t.get_text()]}
(OUT/'geometry_Fig5.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding='utf-8')
prov={'sources':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},
 'rows':len(plot),'local_points':162,'comparator_curve_points':648,'Chen_single_point':1,
 'above_local_envelope':{'utilisation':int(comp.above_util_q50.sum()),'groups':int(comp.above_closed_loop.sum())},
 'bounds_reason':'All observations visible, including utilisation38.813% and realised exceedance20.833%.',
 'envelope_definition':'Best local-predictor outcome across fixed and online points with exceed_rows <= query exceed_rows; independently for each outcome.'}
(OUT/'provenance_Fig5.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
plt.close(fig);doc=fitz.open(OUT/'Fig5.pdf')
for dpi,name in [(160,'preview'),(96,'print_size_96dpi'),(300,'pdf_render_300dpi')]:
    doc[0].get_pixmap(dpi=dpi,alpha=False).save(OUT/f'Fig5_{name}.png')
im=Image.open(OUT/'Fig5_preview.png').convert('RGB');ImageOps.grayscale(im).save(OUT/'Fig5_grayscale.png')
rgb=np.asarray(im)/255;lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
m=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.01182,.04294,.968881]])
sim=np.clip(lin@m.T,0,1);srgb=np.where(sim<=.0031308,12.92*sim,1.055*sim**(1/2.4)-.055)
Image.fromarray(np.uint8(np.clip(srgb*255,0,255))).save(OUT/'Fig5_deuteranopia.png')
print('Built Fig5:162 local +648 comparator curve points +1 Chen reproduction point;0 comparator points above either local envelope.')
