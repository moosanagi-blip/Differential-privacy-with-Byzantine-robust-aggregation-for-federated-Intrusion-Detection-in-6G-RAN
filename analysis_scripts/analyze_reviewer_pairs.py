"""Paired descriptive comparisons; no inference from incomplete triplets."""
from pathlib import Path
import json
import hashlib
import statistics as st

base = Path(__file__).resolve().parents[1]
root = base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26'
summary = json.loads((root/'analysis_initial/summary.json').read_text())
comparisons = []
def compare(label, treatment, reference):
    left = {r['seed']:r for r in summary[treatment]['replicates']}
    right = {r['seed']:r for r in summary[reference]['replicates']}
    pairs = []
    for seed in (42,43,44):
        a,b = left[seed],right[seed]
        pairs.append({'seed':seed, 'treatment_status':a['status'], 'reference_status':b['status'],
            'differences':{k:a['metrics'][k]-b['metrics'][k] for k in a['metrics']}
                if a['status']==b['status']=='completed' else None})
    complete = all(p['differences'] is not None for p in pairs)
    comparisons.append({'label':label, 'treatment':treatment, 'reference':reference,
        'direction':'treatment minus reference', 'pairs':pairs,
        'aggregate':{k:{'mean':st.mean(p['differences'][k] for p in pairs),
                        'sample_sd':st.stdev(p['differences'][k] for p in pairs)}
                     for k in ('accuracy','macro_f1','weighted_f1','fpr')} if complete else None})

for agg in ('multikrum','uniformavg','fedavg'):
    for attack in ('clean','static','rotating'):
        compare(f'Privacy: {agg}, {attack}',f'primary_dp5_{agg}_{attack}',f'primary_nonprivate_{agg}_{attack}')
    for privacy in ('nonprivate','dp5'):
        for attack in ('static','rotating'):
            compare(f'Attack: {privacy}, {agg}, {attack}', f'primary_{privacy}_{agg}_{attack}',f'primary_{privacy}_{agg}_clean')
for privacy in ('nonprivate','dp5'):
    compare(f'Radio masking: {privacy}',f'radio_masked_{privacy}',f'primary_{privacy}_multikrum_clean')
    for f in (0,1,2,4):
        compare(f'Assumed f={f}: {privacy}',f'assumed_f{f}_{privacy}',f'primary_{privacy}_multikrum_static')
    for cap in ('0.5','5'):
        compare(f'Attack cap={cap}: {privacy}',f'cap{cap}_{privacy}',f'primary_{privacy}_multikrum_static')

diagnostics = []
for name,group in summary.items():
    for rep in group['replicates']:
        path = Path(rep['source'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rep['sha256']
        record = json.loads(path.read_text())
        row = {'condition':name,'seed':rep['seed'],'status':rep['status']}
        if rep['status']=='completed':
            result = record['result']
            timeline = record['completed_rounds']
            selected = sum(len(r.get('byz_selected',[])) for r in timeline)
            submitted = sum(len(r.get('byzantine_clients',[])) for r in timeline)
            cm = result['final']['confusion_matrix']
            epsilon = result['final_epsilon_worst_client']
            if record['config']['dp']:
                assert epsilon <= record['config']['target_epsilon'] + 1e-9
            row.update(malicious_selected=selected,malicious_submitted=submitted,
                admission_rate=selected/submitted if submitted else None,
                epsilon_worst=epsilon,
                supported_zero_recall_classes=sum(sum(cm[i])>0 and cm[i][i]==0 for i in range(34)))
        diagnostics.append(row)

out = root/'analysis_paired'
out.mkdir(exist_ok=False)
with (out/'paired_comparisons.json').open('x') as handle:
    json.dump(comparisons,handle,indent=2,allow_nan=False)
with (out/'run_diagnostics.json').open('x') as handle:
    json.dump(diagnostics,handle,indent=2,allow_nan=False)
lines = ['# Paired reviewer comparisons', '',
    'Differences are treatment minus reference, paired by partition/training seed.',
    'Accuracy differences use percentage points; macro-F1 differences use raw units.',
    'SD is the sample standard deviation of the three paired differences, not a confidence interval.',
    'Incomplete comparisons retain individual pairs in JSON but have no aggregate; no significance claim is made.', '',
    '| Comparison | Accuracy difference (pp), mean ± SD | Macro-F1 difference, mean ± SD |',
    '|---|---:|---:|']
for c in comparisons:
    a = c['aggregate']
    acc = f"{100*a['accuracy']['mean']:.3f} ± {100*a['accuracy']['sample_sd']:.3f}" if a else 'Incomplete'
    f1 = f"{a['macro_f1']['mean']:.4f} ± {a['macro_f1']['sample_sd']:.4f}" if a else 'Incomplete'
    lines.append(f"| {c['label']} | {acc} | {f1} |")
(out/'PAIRED_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'{len(comparisons)} paired comparisons; {sum(c["aggregate"] is not None for c in comparisons)} complete; all completed private budgets within target.')
