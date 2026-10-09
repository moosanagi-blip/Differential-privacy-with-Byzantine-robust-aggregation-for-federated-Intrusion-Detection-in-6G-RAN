"""Seed-level descriptive statistics; no training or public diagnostic release."""
from pathlib import Path
import hashlib, json, math, statistics as st
import numpy as np
base=Path(__file__).resolve().parents[1]
root=base/'paper gpt/dp-fl-ids code/results'
out=base/'output/final_gap_statistics_20261006'
out.mkdir(exist_ok=True)
hashes={}
def read(p):
    hashes[str(p.relative_to(base))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def summary(v):
    assert len(v)==3 and all(math.isfinite(x) for x in v)
    return dict(values=v,mean=st.mean(v),sample_sd=st.stdev(v),n_seeds=3)
def audit(r):
    assert r['status']=='completed' and len(r['completed_rounds'])==30
    f=r['result']['final']; cm=np.asarray(f['confusion_matrix'],dtype=float)
    assert cm.shape==(34,34) and np.all(cm>=0)
    tp=np.diag(cm); sup=cm.sum(axis=1); pred=cm.sum(axis=0)
    f1=np.divide(2*tp,sup+pred,out=np.zeros(34),where=(sup+pred)>0)
    for key,value in dict(accuracy=tp.sum()/cm.sum(),macro_f1=f1.mean(),weighted_f1=np.dot(f1,sup)/sup.sum()).items():
        assert math.isclose(f[key],value,rel_tol=1e-10,abs_tol=1e-12),(key,f[key],value)
    for e in r['completed_rounds']:
        assert max(e['epsilon_by_client'].values())<=r['config']['target_epsilon']+1e-9
    return f

displacement=[]
for eps in (1,5,10):
    for seed in (42,43,44):
        p=(root/'boundary_extension_20261001'/f'adaptive_krum_dp5_cap2_s{seed}' if eps==5 else root/'final_gap_20261003'/f'displacement_dp{eps}_s{seed}')/'record.json'
        r=read(p);f=audit(r); c=r['config']
        assert c['attack']=='adaptive_krum' and c['attack_scale']==2 and c['seed']==seed
        medians={k:[] for k in ('perturbation_l2','perturbation_over_honest_radius','perturbation_over_pre_attack_update')}
        for e in r['completed_rounds']:
            clients=e['attack_measurement']['clients'];assert len(clients)==3
            for k in medians:
                v=[x[k] for x in clients]
                assert all(x is not None and math.isfinite(x) and x>=0 for x in v)
                medians[k].append(st.median(v))
        displacement.append(dict(epsilon=eps,seed=seed,**{k:st.median(v) for k,v in medians.items()},
            admission_fraction=sum(len(e['byz_selected']) for e in r['completed_rounds'])/90,
            accuracy=f['accuracy'],macro_f1=f['macro_f1']))
keys=[k for k in displacement[0] if k not in ('epsilon','seed')]
groups={str(eps):{k:summary([r[k] for r in displacement if r['epsilon']==eps]) for k in keys} for eps in (1,5,10)}
contrasts={}
for high,low in ((5,1),(10,5),(10,1)):
    contrasts[f'{high}_minus_{low}']={k:summary([next(r[k] for r in displacement if r['epsilon']==high and r['seed']==s)-next(r[k] for r in displacement if r['epsilon']==low and r['seed']==s) for s in (42,43,44)]) for k in keys}

noise=[]
for seed in (42,43,44):
    folder=root/'final_gap_20261003'/f'noise_selection_dp5_s{seed}'
    r=read(folder/'record.json');audit(r)
    correlations=[];normalised_gaps=[];rescue=harm=all_flips=0
    for e in r['completed_rounds']:
        n=read(folder/f'noise_round_{e["round"]:02d}.json');cs=n['clients']
        assert len(cs)==10 and {c['client'] for c in cs}==set(range(10))
        values=np.array([c['noise_induced_payload_change_l2'] for c in cs])
        selected=np.array([c['selected_actual'] for c in cs],dtype=bool)
        assert selected.sum()==7 and sum(c['selected_all_noise_suppressed'] for c in cs)==7
        assert np.all(np.isfinite(values)) and np.all(values>=0)
        assert {c['client'] for c in cs if c['selected_actual']}==set(e['selected'])
        # Average ranks handle ties; correlation with binary rejection is descriptive.
        ranks=np.array([1+sum(values<v)+.5*(sum(values==v)-1) for v in values])
        if np.std(ranks)>0:correlations.append(float(np.corrcoef(ranks,(~selected).astype(float))[0,1]))
        if values.mean()>0:normalised_gaps.append(float((values[~selected].mean()-values[selected].mean())/values.mean()))
        rescue+=sum(not c['selected_actual'] and c['selected_own_noise_suppressed'] for c in cs)
        harm+=sum(c['selected_actual'] and not c['selected_own_noise_suppressed'] for c in cs)
        all_flips+=sum(c['selected_actual']!=c['selected_all_noise_suppressed'] for c in cs)
    noise.append(dict(seed=seed,mean_round_rank_rejection_correlation=st.mean(correlations),
        defined_correlation_rounds=len(correlations),mean_normalized_rejected_minus_selected_gap=st.mean(normalised_gaps),
        defined_gap_rounds=len(normalised_gaps),own_noise_removal_rescues=rescue,
        own_noise_removal_harms=harm,rescue_fraction_of_90_rejections=rescue/90,
        harm_fraction_of_210_selections=harm/210,net_selection_change_pp=100*(rescue-harm)/300,
        all_noise_removed_flip_fraction=all_flips/300))
noise_groups={k:summary([r[k] for r in noise]) for k in noise[0] if k not in ('seed','defined_correlation_rounds','defined_gap_rounds')}
payload=dict(displacement_seeds=displacement,displacement_groups=groups,paired_epsilon_contrasts=contrasts,
    noise_seeds=noise,noise_groups=noise_groups,source_sha256=hashes,
    scope='Descriptive mean and sample SD across three seeds. Within-run medians first over attackers then rounds. No client/round-level significance testing. Counterfactuals conditional on actual global trajectory, not ordinary SGD or raw Gaussian norm. Diagnostics not DP-protected; local research only.')
(out/'statistics.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
lines=['# Final reviewer-gap statistical analysis','',
    'Local research output. Counterfactual-derived statistics are not covered by the live training privacy guarantee; do not automatically publish this folder.',
    '', '## Method','',
    'Three paired seeds (42/43/44) are the replicates. Summaries are mean ± sample SD, not confidence intervals. Rounds, classes and clients are not independent samples. No significance claims are made with n=3. All 12 input run records (nine new and three epsilon-5 references) passed confusion-matrix-derived accuracy/F1 checks and executed privacy-bound checks.',
    '', 'Displacement: median over three attackers in each round, then median over 30 rounds, then seed-level summaries. Same proxy attack and cap 2 at epsilon 1/5/10. The pre-attack reference already includes DP training noise.',
    '', '## Displacement results','', '| Epsilon | Absolute displacement | Displacement / honest radius | Displacement / own increment | Admission | Macro-F1 |','|---|---|---|---|---|---|']
def fmt(s):return f"{s['mean']:.6g} ± {s['sample_sd']:.6g}"
for eps in (1,5,10):lines.append('| '+str(eps)+' | '+' | '.join(fmt(groups[str(eps)][k]) for k in ('perturbation_l2','perturbation_over_honest_radius','perturbation_over_pre_attack_update','admission_fraction','macro_f1'))+' |')
lines+=['','## Noise-selection results','',
    'Each honest client is replayed with clipped DP-SGD and matched random draws, suppressing the noise contribution. Actual training is unchanged. Replay differences include altered local gradient trajectories. One-client replacements hold the other nine actual payloads fixed.',
    '', '| Statistic | Mean ± sample SD across seeds |','|---|---|']
for k,v in noise_groups.items():lines.append(f'| {k} | {fmt(v)} |')
lines+=['','Rescue means an actually rejected client becomes selected when only its own noise contribution is suppressed; harm is the reverse. Their denominators differ (90 rejected and 210 selected client-rounds per seed). Net selection change uses all 300 client-rounds. All-noise-removed flips compare two seven-client sets and measure membership changes, not utility.',
    '', 'Association between payload displacement and rejection is not, alone, proof of causality. Own-client replacement tests a conditional local intervention, not long-run denoising benefit. The three seeds share a fixed prepared dataset and holdout, limiting generalisation. Paired epsilon differences and all per-seed summaries are in statistics.json.',
    '', 'No new training, manuscript claims, or public-upload changes were made by this analysis.']
(out/'STATISTICAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'displacement':groups,'noise':noise_groups},indent=2))
