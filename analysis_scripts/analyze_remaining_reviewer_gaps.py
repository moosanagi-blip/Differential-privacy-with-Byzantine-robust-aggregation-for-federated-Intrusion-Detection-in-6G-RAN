"""Descriptive reviewer analyses from frozen records; no model training."""
from pathlib import Path
import hashlib
import json
import math
import os
import statistics as st

base=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(base/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=base/'paper gpt/dp-fl-ids code/results'
paper=base/'paper gpt/paper'
out=base/'output/reviewer_gap_analysis_20261003'
out.mkdir(exist_ok=True)
summary=json.loads((root/'reviewer_corrected_2026-09-26/analysis_initial/summary.json').read_text())
sources={}
def read(path):
    sources[str(path.relative_to(base)).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def record(condition,seed):
    rep=next(r for r in summary[condition]['replicates'] if r['seed']==seed)
    path=Path(rep['source'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rep['sha256']
    return read(path)
def aggregate(values):
    return {'mean':st.mean(values),'sample_sd':st.stdev(values)} if len(values)==3 else None
def median(values):
    values=[v for v in values if v is not None]
    return st.median(values) if values else None
def export(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    fig.savefig(paper/f'figures/{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)

# R2.5: ratios and actual admission across the measured strength sweep.
strength=[]
for path in sorted((root/'boundary_extension_20261001').glob('*/record.json')):
    r=read(path);cfg=r['config']; rounds=[]
    for e in r['completed_rounds']:
        d=e['diagnostics']['score_selection'];clients=d['clients']
        honest=[c['score'] for c in clients if not c['malicious']]
        bad=[c['score'] for c in clients if c['malicious']]
        assert len(honest)==7 and len(bad)==3
        h=st.mean(honest);cutoff=d['max_selected_score']
        rounds.append({'round':e['round'],'malicious_mean_score':st.mean(bad),
            'honest_mean_score':h,'score_ratio':st.mean(bad)/h if h else None,
            'median_normalized_margin':median([c['signed_margin']/abs(cutoff) if cutoff else None for c in clients if c['malicious']]),
            'selected_attackers':len(e['byz_selected'])})
    strength.append({'name':cfg['name'],'dp':cfg['dp'],'cap':cfg['attack_scale'],
        'attack':cfg['attack'],'seed':cfg['seed'],'status':r['status'],
        'completed_rounds':len(rounds),'rounds':rounds,
        'median_score_ratio':median([e['score_ratio'] for e in rounds]),
        'median_normalized_margin':median([e['median_normalized_margin'] for e in rounds]),
        'admission_rate':sum(e['selected_attackers'] for e in rounds)/(3*len(rounds)) if rounds else None})
fig,axes=plt.subplots(2,3,figsize=(12,7))
for row,dp in enumerate((False,True)):
    for ai,(attack,label,color) in enumerate([('adaptive_krum','Proxy','#0072B2'),('boundary_krum','Selection-aware','#D55E00')]):
        for ci,cap in enumerate((.5,2.,5.)):
            for si,seed in enumerate((42,43,44)):
                v=next(x for x in strength if (x['dp'],x['attack'],x['cap'],x['seed'])==(dp,attack,cap,seed))
                for col,(key,factor) in enumerate([('admission_rate',100),('median_score_ratio',1),('median_normalized_margin',1)]):
                    ax=axes[row,col];x=ci+(ai-.5)*.24+(si-1)*.035
                    if v['status']=='completed':
                        if v[key] is not None:ax.scatter(x,v[key]*factor,c=color,marker=('o','s','^')[si],s=32,label=label if ci==0 and si==0 else None)
                    else:ax.scatter(x,-.13,transform=ax.get_xaxis_transform(),marker='x',c=color,clip_on=False)
    for col,title in enumerate(('Attacker admission (%)','Median mean-score ratio','Median normalized margin')):
        ax=axes[row,col];ax.set_title(('DP off: ' if not dp else 'Target epsilon 5: ')+title,fontsize=10)
        ax.set_xticks(range(3),['0.5','2','5']);ax.set_xlabel('Search cap',labelpad=16);ax.grid(alpha=.2)
    axes[row,0].set_ylim(-2,104)
axes[0,1].legend(fontsize=8)
fig.suptitle('Markers: seeds 42 / 43 / 44; crosses below axes: numerical failures',fontsize=11)
export(fig,'reviewer_strength_scores')

# R1.6: fix support-based groups before comparing outcomes; include all classes.
reference=record('primary_nonprivate_multikrum_clean',42)['result']['final']
names=reference['class_order'];cm=reference['confusion_matrix'];supports=list(map(sum,cm))
order=sorted(range(34),key=lambda i:(supports[i],names[i]))
minority=order[:math.ceil(34/4)];other=[i for i in range(34) if i not in minority]
comparisons=[]
tests=[('Privacy in clean Multi-Krum','primary_dp5_multikrum_clean','primary_nonprivate_multikrum_clean'),
       ('Static attack under DP','primary_dp5_multikrum_static','primary_dp5_multikrum_clean'),
       ('Rotating attack under DP','primary_dp5_multikrum_rotating','primary_dp5_multikrum_clean')]
for label,treatment,control in tests:
    pairs=[]
    for seed in (42,43,44):
        a=record(treatment,seed);b=record(control,seed)
        assert a['status']==b['status']=='completed'
        for r in (a,b):
            assert r['result']['final']['class_order']==names
            assert list(map(sum,r['result']['final']['confusion_matrix']))==supports
        recalls=[[r['result']['final']['confusion_matrix'][i][i]/supports[i] for i in range(34)] for r in (a,b)]
        delta=[100*(x-y) for x,y in zip(*recalls)]
        low=st.mean(delta[i] for i in minority);rest=st.mean(delta[i] for i in other)
        pairs.append({'seed':seed,'minority_delta_pp':low,'other_delta_pp':rest,
            'minority_minus_other_delta_pp':low-rest,'class_delta_pp':delta,
            'treatment_recall':recalls[0],'control_recall':recalls[1]})
    comparisons.append({'label':label,'treatment':treatment,'control':control,'pairs':pairs,
        'aggregate':{key:aggregate([p[key] for p in pairs]) for key in ('minority_delta_pp','other_delta_pp','minority_minus_other_delta_pp')}})
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,comparison in zip(axes,comparisons):
    for pos,key in enumerate(('minority_delta_pp','other_delta_pp')):
        vals=[p[key] for p in comparison['pairs']]
        for j,v in enumerate(vals):ax.scatter(pos+(j-1)*.055,v,marker=('o','s','^')[j],color=('#0072B2','#D55E00','#009E73')[j])
        ax.errorbar(pos,st.mean(vals),yerr=st.stdev(vals),fmt='_',color='black',capsize=4,markersize=15)
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks([0,1],['Lowest-support\n9 classes','Other\n25 classes'])
    ax.set_title(comparison['label'],fontsize=10);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Treatment minus control mean recall (pp)')
fig.suptitle('Support-defined groups; equal weight per class; points are paired seeds, bars are sample SD',fontsize=10)
export(fig,'reviewer_minority_degradation')

# R2.10: describe selection changes and matched utility; noise components absent.
selection=[];utility=[]
for seed in (42,43,44):
    a=record('primary_dp5_multikrum_clean',seed);b=record('primary_nonprivate_multikrum_clean',seed)
    counts={mode:[0]*10 for mode in ('dp','nonprivate')};overlaps=[]
    for ea,eb in zip(a['completed_rounds'],b['completed_rounds']):
        assert ea['round']==eb['round']
        sa=set(ea['selected']);sb=set(eb['selected']);assert len(sa)==len(sb)==7
        assert not ea['byzantine_clients'] and not eb['byzantine_clients']
        overlaps.append(len(sa&sb)/len(sa|sb))
        for mode,chosen in [('dp',sa),('nonprivate',sb)]:
            for i in chosen:counts[mode][i]+=1
    selection.append({'seed':seed,'selected_rounds_by_client':counts,'mean_round_jaccard':st.mean(overlaps),
                      'round_jaccard':overlaps})
    values={}
    for privacy in ('nonprivate','dp5'):
        mk=record(f'primary_{privacy}_multikrum_clean',seed)['result']['final']
        avg=record(f'primary_{privacy}_uniformavg_clean',seed)['result']['final']
        values[privacy]={metric:mk[metric]-avg[metric] for metric in ('accuracy','macro_f1','fpr')}
    utility.append({'seed':seed,'multikrum_minus_equal_mean':values,
        'dp_minus_nonprivate_selection_effect':{k:values['dp5'][k]-values['nonprivate'][k] for k in values['dp5']}})
utility_aggregate={mode:{k:aggregate([r['multikrum_minus_equal_mean'][mode][k] for r in utility]) for k in ('accuracy','macro_f1','fpr')} for mode in ('nonprivate','dp5')}
fig,axes=plt.subplots(1,2,figsize=(11,3.4),sharey=True)
for ax,mode,title in zip(axes,('nonprivate','dp'),('Clean Multi-Krum: no DP','Clean Multi-Krum: target epsilon 5')):
    matrix=np.array([r['selected_rounds_by_client'][mode] for r in selection])
    im=ax.imshow(matrix,vmin=0,vmax=30,cmap='Blues',aspect='auto')
    for i in range(3):
        for j in range(10):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>18 else 'black')
    ax.set_xticks(range(10));ax.set_yticks(range(3),['42','43','44']);ax.set_xlabel('Client ID');ax.set_title(title,fontsize=10)
axes[0].set_ylabel('Paired seed')
fig.suptitle('Number of selected rounds (out of 30); seven of ten honest clients selected per round',fontsize=10)
export(fig,'reviewer_clean_selection')
payload={'source_sha256':sources,'strength_sweep':strength,'minority_definition':{'method':'Nine classes with lowest held-out support, chosen by support alone; post-hoc descriptive grouping, not preregistered.',
    'minority_classes':[names[i] for i in minority],'class_order':names,'heldout_supports':supports},'minority_comparisons':comparisons,
    'clean_selection':selection,'clean_utility_pairs':utility,'clean_utility_aggregate':utility_aggregate,
    'limitations':'No saved counterfactual noise-free local payloads: cannot identify noise realizations or test preferential rejection by noise magnitude. Achieved attack displacement is unavailable at epsilon 1 and 10. Rounds/classes are not extra independent seeds.'}
(out/'analysis.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
print(json.dumps({'minority_classes':[names[i] for i in minority],
    'minority_results':[{ 'label':c['label'],**c['aggregate']} for c in comparisons],
    'selection_jaccard':[r['mean_round_jaccard'] for r in selection],
    'utility':utility_aggregate},indent=2))
