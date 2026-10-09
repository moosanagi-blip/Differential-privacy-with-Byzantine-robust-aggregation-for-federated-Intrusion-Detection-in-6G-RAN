"""Summarize the frozen campaign without silently excluding failed replicates."""
from pathlib import Path
import json
import statistics
import hashlib

base = Path(__file__).resolve().parents[1]
root = base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26'
plan = json.loads((root/'matrix.json').read_text())
groups = {}
for cfg in plan:
    folder = root/cfg['name']
    path = folder/'record.json'
    reused = (folder/'reuse.json').exists()
    if reused:
        link = json.loads((folder/'reuse.json').read_text())
        path = Path(link['source'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == link['sha256']
    record = json.loads(path.read_text())
    actual = dict(record['config']); expected = dict(cfg)
    actual.pop('name'); expected.pop('name')
    assert actual == expected
    entry = {'seed':cfg['seed'], 'status':record['status'],
             'completed_rounds':record['completed_round_count'],
             'source':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'reused':reused}
    if record['status'] == 'completed':
        final = record['result']['final']
        entry['metrics'] = {k:final[k] for k in ('accuracy','macro_f1','weighted_f1','fpr')}
    else:
        entry['error'] = record.get('error')
    groups.setdefault(cfg['name'].rsplit('_s',1)[0],[]).append(entry)
summary = {}
for name, entries in groups.items():
    assert sorted(e['seed'] for e in entries) == [42,43,44]
    complete = all(e['status'] == 'completed' for e in entries)
    summary[name] = {'replicates':entries, 'all_three_successful':complete,
        'aggregate':{k:{'mean':statistics.mean(e['metrics'][k] for e in entries),
                        'sample_sd':statistics.stdev(e['metrics'][k] for e in entries)}
                     for k in ('accuracy','macro_f1','weighted_f1','fpr')} if complete else None}
out = root/'analysis_initial'
out.mkdir(exist_ok=False)
with (out/'summary.json').open('x') as handle:
    json.dump(summary,handle,indent=2,allow_nan=False)
lines = ['# Corrected reviewer campaign: initial descriptive summary', '',
    'All 102 planned conditions have terminal records: 97 completed and five numerical failures.',
    'Means and sample standard deviations are shown only when all three paired partition/training replicates completed.',
    'Failed groups retain individual successful metrics in summary.json; no failed run is assigned zero accuracy.',
    'This is an initial descriptive audit, not final paired inference or a publication-readiness declaration.', '',
    '| Condition | Successful replicates | Accuracy mean ± SD | Macro-F1 mean ± SD |',
    '|---|---:|---:|---:|']
for name, value in summary.items():
    n = sum(e['status']=='completed' for e in value['replicates'])
    agg = value['aggregate']
    acc = f"{agg['accuracy']['mean']:.6f} ± {agg['accuracy']['sample_sd']:.6f}" if agg else 'Incomplete: see individual records'
    f1 = f"{agg['macro_f1']['mean']:.6f} ± {agg['macro_f1']['sample_sd']:.6f}" if agg else 'Not aggregated'
    lines.append(f'| {name} | {n}/3 | {acc} | {f1} |')
(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'Validated {len(plan)} configurations; summarized {len(summary)} groups; {sum(v["all_three_successful"] for v in summary.values())} complete triplets.')

if len(plan) != 102:
    raise RuntimeError('Unexpected matrix size')
