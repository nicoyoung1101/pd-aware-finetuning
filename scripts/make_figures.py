"""Recreate public aggregate-result figures; no motion data or GPU required."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'assets'
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':'#334155',
                     'text.color':'#17263d','axes.edgecolor':'#d7e0ea','savefig.facecolor':'white'})
COLORS=['#159d95','#e39c46','#6779c6','#ba6591','#61788f']

def rows(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))

is_fine=(ROOT/'results/five_descriptor_paired_effects.csv').exists()
fig=plt.figure(figsize=(16,4.5),facecolor='#0d1c31')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,16);ax.set_ylim(0,4.5);ax.axis('off')
ax.text(.8,3.55,'02 / DOMAIN ADAPTATION' if is_fine else '01 / RESEARCH BENCHMARK',
        color='#64ded0',fontsize=15,weight='bold')
ax.text(.8,2.42,'PD-aware Fine-tuning' if is_fine else 'PD Gait Benchmark',color='white',fontsize=37,weight='bold')
ax.text(.8,1.7,'Clinically relevant motion, with sparse sensors' if is_fine else 'Beyond pose accuracy: preserving Parkinsonian gait',color='#bdd0e2',fontsize=17)
ax.text(.8,.65,'23 participants  /  781 walks  /  0.48% trainable' if is_fine else '5 models  /  6 virtual IMUs  /  2 cohorts',color='#dbe7ef',fontsize=14)
points=np.array([[12.7,3.45],[12.7,2.8],[12.2,2.35],[13.2,2.35],[11.8,1.75],[13.6,1.75],[12.7,1.9],[12.2,1.2],[13.2,1.2],[11.9,.65],[13.5,.65]])
edges=[(0,1),(1,2),(1,3),(2,4),(3,5),(1,6),(6,7),(6,8),(7,9),(8,10)]
for i,j in edges:ax.plot(points[[i,j],0],points[[i,j],1],color='#557088',lw=4,zorder=1)
for i in [0,4,5,6,9,10]:
    x,y=points[i];ax.scatter(x,y,s=80,color='#64ded0',zorder=2);ax.scatter(x,y,s=310,facecolors='none',edgecolors='#1f665f',lw=1,zorder=1)
fig.savefig(ASSETS/'banner.png',dpi=100,facecolor=fig.get_facecolor());plt.close(fig)

if is_fine:
    data=rows('five_descriptor_paired_effects.csv')
    descriptors=['Lower-body velocity','Foot lift','Lower-body RoM','Joint velocity','Leg swing']
    assert len(data)==10
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.2),sharey=True)
    for ax,sensor,title in zip(axes,['Six','Five'],['Six IMUs','Head-free five IMUs']):
        subset={r['descriptor']:r for r in data if r['sensors']==sensor}
        for i,d in enumerate(descriptors):
            r=subset[d];v=float(r['clinical_minus_pose']);lo=float(r['ci_low']);hi=float(r['ci_high'])
            assert lo<=v<=hi and int(r['n_subjects'])==23 and int(r['n_walk_pairs'])==781
            color='#159d95' if hi<0 else '#c2685e' if lo>0 else '#778597'
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,markersize=8,capsize=4,lw=2)
        ax.axvline(0,color='#788699',lw=1,ls='--');ax.set_xlim(-.065,.065)
        ax.set_yticks(range(5),descriptors);ax.grid(axis='x',alpha=.15);ax.set_title(title,fontsize=16,weight='bold',pad=15)
        ax.set_xlabel('Clinical-aware minus pose-only distortion')
    axes[0].invert_yaxis()
    fig.suptitle('Targeted gait-feature gains, with a foot-lift tradeoff',fontsize=20,weight='bold',y=.98)
    fig.text(.5,.025,'23 participants · subject-equal means · nominal 95% clustered bootstrap CIs · post-hoc benchmark-aligned analysis',ha='center',fontsize=10,color='#607084')
    fig.tight_layout(rect=[0,.065,1,.92]);fig.savefig(ASSETS/'finetuning-results.png',dpi=160);plt.close(fig)
else:
    fidelity=rows('pose_fidelity.csv');gait=rows('gait_preservation.csv')
    models=['TransPose','PIP raw','DynaIP','PNP','TIP']
    fig,axes=plt.subplots(1,3,figsize=(14.4,5))
    for model,color in zip(models,COLORS):
        sets=[[r for r in fidelity if r['model']==model and r['metric']=='MPJPE'],
              [r for r in gait if r['model']==model and r['descriptor']=='Joint velocity ratio'],
              [r for r in gait if r['model']==model and r['descriptor']=='Leg swing ratio']]
        for ax,selected in zip(axes,sets):
            assert len(selected)==1
            ax.plot(range(3),[float(selected[0][f'U{i}']) for i in range(3)],'o-',label=model,color=color,lw=2,markersize=6)
    for ax,title,ylabel in zip(axes,['Pose fidelity','Joint velocity preservation','Leg-swing preservation'],['Root-relative MPJPE (mm)','Prediction / GT ratio','Prediction / GT ratio']):
        ax.set_title(title,fontsize=14,weight='bold',pad=12);ax.set_ylabel(ylabel)
        ax.set_xticks(range(3),['U0','U1','U2']);ax.set_xlabel('MDS-UPDRS gait');ax.grid(alpha=.15)
    for ax in axes[1:]:ax.axhline(1,color='#8b99a8',ls='--',lw=1)
    fig.suptitle('Geometric accuracy does not fully describe gait preservation',fontsize=20,weight='bold',y=.99)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,.03),frameon=False)
    fig.text(.5,.015,'Reported severity-stratified snapshot · BMCLab · virtual six-IMU inputs · ratio reference = 1',ha='center',fontsize=10,color='#607084')
    fig.tight_layout(rect=[0,.13,1,.92]);fig.savefig(ASSETS/'benchmark-results.png',dpi=160);plt.close(fig)
print('Regenerated figures:',ASSETS)
