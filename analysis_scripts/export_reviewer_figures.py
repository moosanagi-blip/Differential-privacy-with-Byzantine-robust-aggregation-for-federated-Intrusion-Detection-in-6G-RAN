"""Export PNG/SVG comparison figures; do not generate a manuscript PDF."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

base=Path(__file__).resolve().parents[1]
root=base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26'
data=json.loads((root/'analysis_initial/summary.json').read_text())
out=base/'output/reviewer_figures_20260930'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
colors=['#0072B2','#D55E00','#009E73']

def panel(ax,names,labels,metric,title):
    for x,name in enumerate(names):
        group=data[name]
        for j,r in enumerate(group['replicates']):
            if r['status']=='completed':
                ax.scatter(x+(j-1)*.14,r['metrics'][metric],color=colors[j],s=30,zorder=3)
            else:
                ax.scatter(x+(j-1)*.14,-.075,color=colors[j],marker='x',s=45,zorder=3)
        if group['aggregate']:
            a=group['aggregate'][metric]
            ax.errorbar(x,a['mean'],yerr=a['sample_sd'],color='black',fmt='_',
                        markersize=16,capsize=4,lw=1.2,zorder=2)
    ax.set_xticks(range(len(labels)),labels)
    ax.set_ylim(-.12,1.04)
    ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.axhline(-.035,color='.7',lw=.6)
    ax.set_ylabel('Accuracy' if metric=='accuracy' else 'Macro-F1')
    ax.set_title(title)
    ax.grid(axis='y',alpha=.2)

def save(fig,name,title):
    fig.suptitle(title,fontsize=15,y=.99)
    handles=[Line2D([],[],marker='o',ls='',color=c,label=f'Seed {s}') for c,s in zip(colors,(42,43,44))]
    handles += [Line2D([],[],color='black',marker='_',label='Mean ± sample SD (3/3 only)'),
                Line2D([],[],color='black',marker='x',ls='',label='Numerical failure (below axis)')]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.012),frameon=False)
    fig.tight_layout(rect=(0,.105,1,.955))
    fig.savefig(out/f'{name}.png',dpi=300)
    fig.savefig(out/f'{name}.svg')
    plt.close(fig)

fig,axes=plt.subplots(3,2,figsize=(12,11))
for row,agg in enumerate(('multikrum','uniformavg','fedavg')):
    names=[f'primary_{p}_{agg}_{a}' for p in ('nonprivate','dp5') for a in ('clean','static','rotating')]
    labels=['No DP\nclean','No DP\nstatic','No DP\nrotating','DP ε=5\nclean','DP ε=5\nstatic','DP ε=5\nrotating']
    for col,metric in enumerate(('accuracy','macro_f1')):
        panel(axes[row,col],names,labels,metric,{'multikrum':'Multi-Krum (assumed f=3)','uniformavg':'Equal-client mean','fedavg':'Sample-weighted FedAvg'}[agg])
save(fig,'primary_comparisons','Corrected reviewer experiments: full held-out evaluation')

fig,axes=plt.subplots(2,3,figsize=(15,8))
for row,p in enumerate(('nonprivate','dp5')):
    title='No DP' if p=='nonprivate' else 'DP target ε=5'
    for col,(names,labels,label) in enumerate([
        ([f'primary_{p}_multikrum_clean',f'radio_masked_{p}'],['Radio retained','Radio masked'],'Clean feature ablation'),
        ([f'assumed_f{f}_{p}' if f!=3 else f'primary_{p}_multikrum_static' for f in range(5)],['0','1','2','3','4'],'Assumed attacker count; true count=3'),
        ([f'cap0.5_{p}',f'primary_{p}_multikrum_static',f'cap5_{p}'],['0.5','2','5'],'Adaptive search cap (not achieved strength)')]):
        if col==2:
            label='Adaptive search cap\n(not achieved strength)'
        panel(axes[row,col],names,labels,'macro_f1',f'{title}\n{label}')
save(fig,'ablation_sensitivity','Multi-Krum: feature and attack sensitivity')

names=['epsilon1_static','primary_dp5_multikrum_static','epsilon10_static']
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,metric in zip(axes,('accuracy','macro_f1')):
    panel(ax,names,['ε=1','ε=5','ε=10'],metric,'Static adaptive attack; search cap=2')
save(fig,'privacy_budget','Privacy-budget sensitivity: Multi-Krum, three attackers')

diag=json.loads((root/'analysis_paired/run_diagnostics.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5))
names=['primary_dp5_multikrum_static','primary_dp5_multikrum_rotating',
       'cap0.5_dp5','cap5_dp5']
for x,name in enumerate(names):
    rows=sorted([r for r in diag if r['condition']==name],key=lambda r:r['seed'])
    for j,r in enumerate(rows):
        assert r['status']=='completed' and r['malicious_submitted']==90
        axes[0].scatter(x+(j-1)*.14,100*r['admission_rate'],color=colors[j],s=35)
        axes[1].scatter(x+(j-1)*.14,r['supported_zero_recall_classes'],color=colors[j],s=35)
for ax in axes:
    ax.set_xticks(range(4),['Static\ncap=2','Rotating\ncap=2','Static\ncap=0.5','Static\ncap=5'])
    ax.grid(axis='y',alpha=.2)
axes[0].set(ylabel='Malicious submissions selected (%)',ylim=(-3,104),title='Actual selection across 30 rounds')
axes[1].set(ylabel='Supported classes with zero recall',ylim=(-1,35),title='Full-test class failures (34 classes)')
fig.suptitle('Private Multi-Krum (target ε=5): attack admission and class failures',fontsize=14)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=f'Seed {s}') for c,s in zip(colors,(42,43,44))],loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,.09,1,.94))
for ext in ('png','svg'): fig.savefig(out/f'attack_admission_classes.{ext}',dpi=300)
plt.close(fig)

# All fine classes, matched private/non-private clean controls and private attacks.
conditions=['primary_nonprivate_multikrum_clean','primary_dp5_multikrum_clean',
            'primary_dp5_multikrum_static','primary_dp5_multikrum_rotating']
columns=[]; ticks=[]; class_names=None; supports=None
for name,label in zip(conditions,('Clean/no DP','Clean/DP5','Static/DP5','Rotating/DP5')):
    for rep in data[name]['replicates']:
        assert rep['status']=='completed'
        final=json.loads(Path(rep['source']).read_text())['result']['final']
        cm=np.asarray(final['confusion_matrix'])
        support=cm.sum(axis=1)
        if class_names is None: class_names=final['class_order']; supports=support
        assert final['class_order']==class_names and np.array_equal(support,supports)
        columns.append(np.divide(np.diag(cm),support,out=np.zeros(34),where=support>0))
        ticks.append(f'{label}\ns{rep["seed"]}')
fig,ax=plt.subplots(figsize=(13,12))
im=ax.imshow(np.asarray(columns).T,vmin=0,vmax=1,cmap='viridis',aspect='auto')
ax.set_yticks(range(34),[f'{name} (n={int(n):,})' for name,n in zip(class_names,supports)],fontsize=8)
ax.set_xticks(range(12),ticks,rotation=55,ha='right',fontsize=8)
for boundary in (2.5,5.5,8.5): ax.axvline(boundary,color='white',lw=2)
fig.colorbar(im,ax=ax,label='Recall',fraction=.025,pad=.02)
ax.set_title('Multi-Krum: all-class recall on the identical held-out population\nEach column is one replicate; no averaging across seeds')
fig.tight_layout()
for ext in ('png','svg'): fig.savefig(out/f'all_class_recall.{ext}',dpi=300)
plt.close(fig)
print(out)
