"""Read-only integrity check of completed reviewer records."""
from pathlib import Path
import hashlib
import json
import math

base = Path(__file__).resolve().parents[1]
root = base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26'
checked = 0
failures = []
for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    path = folder/'record.json'
    if (folder/'reuse.json').exists():
        reuse = json.loads((folder/'reuse.json').read_text())
        path = Path(reuse['source'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reuse['sha256']
    if not path.exists():
        continue
    record = json.loads(path.read_text())
    if record['status'] != 'completed':
        assert record.get('final') is None
        failures.append({'name':folder.name, 'status':record['status'],
                         'completed_rounds':record['completed_round_count']})
        continue
    assert record['completed_round_count'] == 30
    final = record['result']['final']
    cm = final['confusion_matrix']
    assert len(cm) == 34 and all(len(row) == 34 for row in cm)
    assert all(isinstance(v,int) and v >= 0 for row in cm for v in row)
    total = sum(map(sum,cm))
    assert total == final['evaluated_rows'] == 4673255
    accuracy = sum(cm[i][i] for i in range(34))/total
    assert math.isclose(accuracy, final['accuracy'], abs_tol=1e-12)
    for key in ('accuracy','macro_f1','weighted_f1','fpr'):
        assert math.isfinite(final[key]) and 0 <= final[key] <= 1
    f1 = []
    for i in range(34):
        denom = sum(cm[i]) + sum(row[i] for row in cm)
        f1.append(2*cm[i][i]/denom if denom else 0.)
    assert math.isclose(sum(f1)/34, final['macro_f1'], abs_tol=1e-12)
    supports = list(map(sum, cm))
    assert math.isclose(sum(f1[i]*supports[i] for i in range(34))/total,
                        final['weighted_f1'], abs_tol=1e-12)
    names = final['class_order']
    assert len(names) == len(set(names)) == 34
    benign = names.index('BenignTraffic')
    assert math.isclose(1-cm[benign][benign]/supports[benign],final['fpr'],abs_tol=1e-12)
    for i,name in enumerate(names):
        per = final['per_class'][name]
        assert per['support'] == supports[i]
        assert per['predicted_count'] == sum(row[i] for row in cm)
        assert per['true_positive'] == cm[i][i]
    events = record['completed_rounds']
    assert [e['round'] for e in events] == list(range(1,31))
    for event in events:
        selected = set(event['selected'])
        malicious = set(event['byzantine_clients'])
        assert set(event['byz_selected']) == selected & malicious
        assert len(event['selected']) == len(selected)
        diagnostic = event['diagnostics']
        assert diagnostic['malicious_selected_count'] == len(selected & malicious)
        assert diagnostic['malicious_participant_count'] == len(malicious)
    if record['config']['dp']:
        result = record['result']
        spent = max(result['privacy_accounting']['epsilon_by_client'].values())
        assert math.isclose(spent,result['final_epsilon_worst_client'],abs_tol=1e-12)
        assert spent <= record['config']['target_epsilon'] + 1e-9
    checked += 1
print(json.dumps({'successful_records_checked':checked, 'failures':failures,
                  'checks':'30 ordered rounds, full holdout, accuracy, macro/weighted F1, benign FPR, all class counts, selection flags, executed privacy maxima; reused SHA-256'},indent=2))
