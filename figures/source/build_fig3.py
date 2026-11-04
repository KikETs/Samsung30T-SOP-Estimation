"""Reproducible paired group results from the manuscript's named source CSV."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz
from PIL import Image, ImageOps
from style import STYLE

OUT=Path(__file__).resolve().parents[1]
PROJECT=OUT.parent
DATA=PROJECT/'paper_package_lsad_20261013_r3/data/groups_tau5.csv'
TABLE=PROJECT/'tables/csv/T3_group_results.csv'
W,H=190,125
blue=STYLE['online_scale_adaptation'];red=STYLE['fixed_scale']
grey=STYLE['reference'];ink=STYLE['text']
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'text.color':ink,
 'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':True,
 'axes.linewidth':.7,'xtick.major.width':.7,'xtick.major.size':2.5,
 'mathtext.fontset':'custom','mathtext.rm':'Cambria','mathtext.it':'Cambria:italic',
 'mathtext.bf':'Cambria:bold','mathtext.fallback':None})
df=pd.read_csv(DATA)
required=['arm','evaluation_group','direction','temp_C','horizon_s','n','exceed',
          'util_q50','passed','pass_original','ratio_q50']
assert all(c in df for c in required),set(required)-set(df.columns)
assert len(df)==30 and set(df.arm)=={'Ls B0','Ls+AD'}
keys=['evaluation_group','direction','temp_C','horizon_s']
order=[(c,d,t,10.) for c in ['A1','A17'] for d in ['charge','discharge'] for t in [10,25]]
order += [('B','discharge',t,h) for t,h in [(-20,10),(-10,10),(0,10),(10,10),(25,2),(25,10),(40,10)]]
arms={a:df.loc[df.arm.eq(a)].set_index(keys).loc[order].reset_index() for a in ['Ls B0','Ls+AD']}
table=pd.read_csv(TABLE).iloc[:15]
summary={};records=[]
ys=np.arange(15,dtype=float);ys[4:]+=.65;ys[8:]+=.65
for arm,prefix,expected in [('Ls B0','fixed',(8,6,7)),('Ls+AD','adapted',(11,4,28))]:
    a=arms[arm]
    assert len(a)==15 and a.n.sum()==672 and not a[keys].duplicated().any()
    assert np.isfinite(a[['util_q50','ratio_q50','n','exceed']]).all().all()
    assert a.n.gt(0).all() and a.exceed.between(0,a.n).all()
    assert a.util_q50.between(0,1).all()
    rate=a.exceed/a.n
    assert (a.passed.eq((a.util_q50>=.8)&(rate<=.1))).all()
    assert (a.pass_original.eq((a.exceed==0)&(a.ratio_q50>=.8))).all()
    assert (a.interventions==a.exceed).all() and np.allclose(a.int_rate,rate)
    counts=(int(a.passed.sum()),int(a.pass_original.sum()),int(a.exceed.sum()))
    assert counts==expected,(arm,counts)
    for col,tcol in [('evaluation_group','cell'),('direction','direction'),('temp_C','temperature_C'),('horizon_s','pulse_duration_s'),('n','rows')]:
        assert a[col].tolist()==table[tcol].tolist(),(col,tcol)
    assert a.exceed.tolist()==table[f'{prefix}_exceedances'].tolist()
    assert np.allclose(np.round(a.util_q50*100,1),table[f'{prefix}_median_utilisation_pct'])
    assert np.allclose(np.round(rate*100,1),table[f'{prefix}_exceedance_rate_pct'])
    assert a.passed.map({True:'Pass',False:'Fail'}).tolist()==table[f'{prefix}_util_exc'].tolist()
    assert a.pass_original.map({True:'Pass',False:'Fail'}).tolist()==table[f'{prefix}_strict'].tolist()
    summary[arm]={'groups':15,'rows':int(a.n.sum()),'passing_util_exc':counts[0],'passing_strict':counts[1],'exceedances':counts[2]}
    for i,row in a.iterrows():
        records.append({'arm':arm,'row_index':i,'cell':row.evaluation_group,'direction':row.direction,
          'temperature_C':int(row.temp_C),'duration_s':float(row.horizon_s),'n':int(row.n),
          'median_utilisation_pct':float(row.util_q50*100),'exceedance_pct':float(100*row.exceed/row.n),
          'exceedances':int(row.exceed),'passed':bool(row.passed),'strict':bool(row.pass_original),
          'row_center':float(ys[i]),'display_y':float(ys[i]+(-.12 if arm=='Ls B0' else .12)),
          'overflow':bool(100*row.exceed/row.n>30)})
ad=arms['Ls+AD']
assert ad.iloc[1].exceed==8 and ad.iloc[1].n==73
assert [round(ad.iloc[i].util_q50*100,1) for i in [1,8,9]]==[89.5,73.7,79.9]

fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
def panel(left,width):
    ax=fig.add_axes([left/W,24/H,width/W,80/H])
    ax.set_ylim(15.9,-.6);ax.set_yticks([])
    for s in ['top','right']:ax.spines[s].set_visible(False)
    ax.tick_params(axis='x',direction='out',labelsize=8,pad=3)
    return ax
a=panel(55,43);b=panel(107,36);c=panel(152,36)
a.set_xlim(70,101);a.set_xticks([70,80,90,100])
b.set_xlim(-1,31);b.set_xticks([0,10,20,30])
b.spines['left'].set_position(('data',0));b.spines['left'].set_linewidth(1)
b.spines['left'].set_color('black');b.spines['bottom'].set_bounds(0,30)
c.set_xlim(-.5,3.85);c.axis('off')
a.axvspan(80,101,color=grey,alpha=.14,lw=0,zorder=0)
b.axvspan(0,10,color=grey,alpha=.14,lw=0,zorder=0)
for ax,val in [(a,80),(b,10)]:ax.axvline(val,color=grey,lw=.8,ls=(0,(3,2)),zorder=1)

def ft(x,y,value,size=8.5,ha='center',**kwargs):
    return fig.text(x/W,1-y/H,value,ha=ha,va='center',fontsize=size,**kwargs)
for x,label in [(55,'(a)'),(107,'(b)'),(152,'(c)')]:ft(x,4,label,10,ha='left',weight='bold')
handles=[Line2D([],[],ls='',marker='s',ms=4,mfc='white',mec=red,mew=1,label='Fixed'),
         Line2D([],[],ls='',marker='o',ms=4,mfc=blue,mec=blue,label='Online'),
         Line2D([],[],ls='',marker='>',ms=4.3,mfc=ink,mec=ink,mew=.7,label='above 30%'),
         Line2D([],[],ls='',marker='s',ms=5,mfc=ink,mec=ink,mew=.7,label='Pass'),
         Line2D([],[],ls='',marker='s',ms=5,mfc='white',mec=grey,mew=1,label='Fail')]
method_legend=fig.legend(handles=handles[:3],loc='lower center',bbox_to_anchor=(99/W,.015),ncol=3,
           frameon=False,fontsize=8.5,handletextpad=.5,columnspacing=2,borderaxespad=0)
verdict_legend=fig.legend(handles=handles[3:],loc='lower center',bbox_to_anchor=(170/W,.015),ncol=2,
           frameon=False,fontsize=8.5,handletextpad=.5,columnspacing=1.5,borderaxespad=0)
for ax,x,label in [(a,80,'80% threshold'),(b,10,'10% threshold')]:
    px=ax.transData.transform((x,0))[0]/fig.bbox.width*W
    ft(px,16,label,8)

def page_y(row):
    return H*(1-a.transData.transform((70,row))[1]/fig.bbox.height)
for i,row in arms['Ls B0'].iterrows():
    temp=str(int(row.temp_C)).replace('-','−')
    ft(39.5,page_y(ys[i]),f'{temp} °C',8.5,ha='right')
    ft(51.8,page_y(ys[i]),f'({int(row.n)})',8.5,ha='right')
    if row.horizon_s==2:ft(44,page_y(ys[i]),'2 s',8)
direction_groups=[('Charge',0,1),('Discharge',2,3),
                  ('Charge',4,5),('Discharge',6,7),('Discharge',8,14)]
for direction,start,end in direction_groups:
    ft(28.5,(page_y(ys[start])+page_y(ys[end]))/2,direction,10,ha='right')
for name,start,end in [('A1',0,3),('A17',4,7),('B',8,14)]:
    yt,yb=page_y(ys[start]-.38),page_y(ys[end]+.38)
    fig.add_artist(Line2D(np.array([11,10,10,11])/W,1-np.array([yt,yt,yb,yb])/H,
                         transform=fig.transFigure,color=grey,lw=.8))
    ft(4.8,(yt+yb)/2,name,8.5)

for i,y in enumerate(ys):
    fixed=arms['Ls B0'].iloc[i];adapted=arms['Ls+AD'].iloc[i]
    for ax,col in [(a,'util_q50'),(b,'rate')]:
        vals=[float(r.util_q50*100 if col=='util_q50' else 100*r.exceed/r.n) for r in [fixed,adapted]]
        clipped=[min(v,30) if col=='rate' else v for v in vals]
        ax.plot(clipped,[y-.12,y+.12],color=grey,lw=.6,zorder=2)
        for v,yy,color,marker,fill in zip(vals,[y-.12,y+.12],[red,blue],['s','o'],['white',blue]):
            overflow=col=='rate' and v>30
            ax.plot(30 if overflow else v,yy,marker='>' if overflow else marker,
                    ms=4.3 if overflow else 4,mfc=fill,mec=color,mew=1 if color==red else .7,
                    ls='',zorder=4,clip_on=False)
        if col=='rate' and max(vals)>30:
            assert all(abs(v-200/3)<1e-9 for v in vals)
            ax.text(31.3,y,'66.7 %',fontsize=8,ha='left',va='center',color=ink,clip_on=False)

cols=[0,1,2.35,3.35]
for i,y in enumerate(ys):
    for x,arm,key,color in [(cols[0],'Ls B0','passed',red),(cols[1],'Ls B0','pass_original',red),
                            (cols[2],'Ls+AD','passed',blue),(cols[3],'Ls+AD','pass_original',blue)]:
        passed=bool(arms[arm].iloc[i][key])
        c.plot(x,y,ls='',marker='s',ms=5,mfc=color if passed else 'white',
               mec=color if passed else grey,mew=.7 if passed else 1,clip_on=False)
def cx(x):return c.transData.transform((x,0))[0]/fig.bbox.width*W
for x,s in [(.5,'Fixed'),(2.85,'Online')]:ft(cx(x),12,s,8)
for x,s in zip(cols,['Joint','Strict']*2):ft(cx(x),16,s,7)
for x,n in zip(cols,[8,6,11,4]):ft(cx(x),105,f'{n}/15',8.5)
ft(76.5,112,'Median utilisation (%)',8.5)
ft(125,112,'Exceedance rate (%)',8.5)
ft(170,112,'Passing groups',8.5)

for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Fig3.{ext}',dpi=600,facecolor='white')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
texts=[]
for t in fig.findobj(match=matplotlib.text.Text):
    if t.get_visible() and t.get_text():
        texts.append({'text':t.get_text(),'size_pt':t.get_fontsize(),
                      'bbox_px':list(t.get_window_extent(renderer).bounds)})
(OUT/'geometry_Fig3.json').write_text(json.dumps({'texts':texts,'figure_pixels':list(fig.bbox.bounds),
 'axes':{'a':list(a.get_position().bounds),'b':list(b.get_position().bounds),'c':list(c.get_position().bounds)},
 'row_positions':ys.tolist(),
 'legend_bounds_px':{'methods':list(method_legend.get_window_extent(renderer).bounds),
                     'verdicts':list(verdict_legend.get_window_extent(renderer).bounds)},
 'label_columns_mm':{'direction_right':28.5,'temperature_right':39.5,'count_right':51.8,'duration_center':44},
 'direction_groups':[{'label':s,'first_row':i,'last_row':j,'center_page_y_mm':(page_y(ys[i])+page_y(ys[j]))/2}
                     for s,i,j in direction_groups]},ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(records).to_csv(OUT/'Fig3_plot_data.csv',index=False)
sources=[DATA,TABLE,PROJECT/'manuscript.tex']
(OUT/'provenance_Fig3.json').write_text(json.dumps({'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
 'summary':summary,'table3_comparison':'All 30 rows match rounded values, counts and verdicts.',
 'overflow':'Two 66.6666666667% values drawn at 30% with triangles; above 30% legend identifies truncation, exact values retained in caption/data.',
 'row_offsets':{'fixed':-.12,'adapted':.12},'value_jitter':False,'data_precision':'full source precision'},indent=2),encoding='utf-8')
plt.close(fig)
doc=fitz.open(OUT/'Fig3.pdf')
for dpi,name in [(160,'preview'),(96,'print_size_96dpi'),(300,'pdf_render_300dpi')]:
    doc[0].get_pixmap(dpi=dpi,alpha=False).save(OUT/f'Fig3_{name}.png')
im=Image.open(OUT/'Fig3_preview.png').convert('RGB')
ImageOps.grayscale(im).save(OUT/'Fig3_grayscale.png')
rgb=np.asarray(im)/255;lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
m=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.01182,.04294,.968881]])
sim=np.clip(lin@m.T,0,1);srgb=np.where(sim<=.0031308,12.92*sim,1.055*sim**(1/2.4)-.055)
Image.fromarray(np.uint8(np.clip(srgb*255,0,255))).save(OUT/'Fig3_deuteranopia.png')
print(json.dumps(summary,indent=2))
