"""Export failure-aware extension summaries and scientific figures."""
from pathlib import Path
import json
import statistics as st
import os
base = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(base/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = base/'paper gpt/dp-fl-ids code/results/boundary_extension_20261001'
output = base/'output/boundary_extension_figures'
output.mkdir(parents=True, exist_ok=True)
records = {r['config']['name']: r for p in root.glob('*/record.json') for r in [json.loads(p.read_text())]}
assert len(records) == 36
rows = []
groups = []
for dp in (False, True):
    for cap in (.5, 2., 5.):
        for attack in ('adaptive_krum', 'boundary_krum'):
            subset = [r for r in records.values() if r['config']['dp']==dp and r['config']['attack_scale']==cap and r['config']['attack']==attack]
            assert len(subset)==3
            good = [r for r in subset if r['status']=='completed']
            group = dict(dp=dp, cap=cap, attack=attack, successful=len(good), aggregate=None)
            if len(good)==3:
                group['aggregate'] = {k: {'mean':st.mean(r['result']['final'][k] for r in good), 'sd':st.stdev(r['result']['final'][k] for r in good)} for k in ('accuracy','macro_f1','fpr')}
            groups.append(group)
            for r in subset:
                rounds = r['completed_rounds']
                clients = [c for e in rounds for c in e['attack_measurement']['clients']]
                row = dict(name=r['config']['name'], dp=dp, cap=cap, attack=attack, seed=r['config']['seed'], status=r['status'], completed_rounds=len(rounds))
                row['metrics'] = {k:r['result']['final'][k] for k in ('accuracy','macro_f1','fpr')} if r['status']=='completed' else None
                row['median_displacement_over_radius'] = st.median(c['perturbation_over_honest_radius'] for c in clients if c['perturbation_over_honest_radius'] is not None)
                row['selected_attackers'] = sum(len(e['byz_selected']) for e in rounds)
                row['attacker_opportunities'] = 3*len(rounds)
                row['fallback_rounds'] = sum(not e['attack_measurement']['search']['feasible_candidate_found'] for e in rounds) if attack=='boundary_krum' else None
                rows.append(row)
pairs=[]
for dp in (False, True):
    for cap in (.5,2.,5.):
        paired=[]
        for seed in (42,43,44):
            a,b=[next(r for r in rows if r['dp']==dp and r['cap']==cap and r['seed']==seed and r['attack']==attack) for attack in ('adaptive_krum','boundary_krum')]
            paired.append(dict(seed=seed, delta_macro_f1=b['metrics']['macro_f1']-a['metrics']['macro_f1'] if a['metrics'] and b['metrics'] else None))
        values=[p['delta_macro_f1'] for p in paired if p['delta_macro_f1'] is not None]
        pairs.append(dict(dp=dp,cap=cap,pairs=paired,mean=st.mean(values) if len(values)==3 else None,sd=st.stdev(values) if len(values)==3 else None))
(output/'summary.json').write_text(json.dumps(dict(groups=groups,runs=rows,paired_comparisons=pairs),indent=2,allow_nan=False))
for metric, ylabel, filename in [('macro_f1','Full-test macro-F1','boundary_utility'),('median_displacement_over_radius','Median displacement / honest RMS radius','boundary_displacement')]:
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,dp in zip(axes,(False,True)):
        for ai,(attack,label,color) in enumerate([('adaptive_krum','Percentile proxy','#1764ab'),('boundary_krum','Selection-aware grid','#d66a13')]):
            for ci,cap in enumerate((.5,2.,5.)):
                selected=sorted([r for r in rows if r['dp']==dp and r['cap']==cap and r['attack']==attack],key=lambda r:r['seed'])
                x=ci+(ai-.5)*.22
                for si,r in enumerate(selected):
                    valid=r['status']=='completed'
                    y=r['metrics'][metric] if metric=='macro_f1' and valid else r.get(metric)
                    if valid:
                        ax.scatter(x+(si-1)*.035,y,color=color,marker=('o','s','^')[si],s=35,label=label if ci==0 and si==0 else None)
                    else:
                        ax.scatter(x+(si-1)*.035,-.09,transform=ax.get_xaxis_transform(),marker='x',color=color,clip_on=False)
        ax.set_xticks(range(3),['0.5','2','5']); ax.set_xlabel('Declared search cap')
        ax.set_title('Target epsilon 5' if dp else 'No DP');ax.grid(alpha=.2)
        if metric!='macro_f1':ax.set_ylim(bottom=0)
    axes[0].set_ylabel(ylabel);axes[1].legend(fontsize=8)
    fig.suptitle('Markers: seeds 42, 43, 44; crosses below axis: numerical failures',fontsize=10)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/f'{filename}.{ext}',dpi=300,bbox_inches='tight')
    fig.savefig(base/f'paper gpt/paper/figures/{filename}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
lines=[r'\begin{table}[p]',r'\centering\begingroup\color{blue}\scriptsize',r'\caption{Extension results. Mean $\pm$ sample SD over three seeds; dashes indicate incomplete triplets.}',r'\label{tab:boundary-extension}',r'\begin{tabular}{llcrrr}',r'\toprule',r'Privacy / cap & Attack & Success & Accuracy (\%) & Macro-F1 & FPR (\%) \\',r'\midrule']
for g in groups:
    vals=[f"${g['aggregate'][k]['mean']*factor:.{6 if k=='macro_f1' else 3}f}\\pm{g['aggregate'][k]['sd']*factor:.{6 if k=='macro_f1' else 3}f}$" if g['aggregate'] else '---' for k,factor in [('accuracy',100),('macro_f1',1),('fpr',100)]]
    lines.append(f"{'DP5' if g['dp'] else 'No DP'} / {g['cap']:g} & {'Proxy' if g['attack']=='adaptive_krum' else 'Grid'} & {g['successful']}/3 & "+' & '.join(vals)+r' \\')
lines += [r'\bottomrule\end{tabular}\endgroup',r'\end{table}']
(base/'paper gpt/paper/generated/boundary_extension_table.tex').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(groups=groups,paired_comparisons=pairs),indent=2))
