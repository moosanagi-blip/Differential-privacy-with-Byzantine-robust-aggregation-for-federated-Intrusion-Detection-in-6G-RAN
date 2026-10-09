"""Read-only checks for completed targeted extension runs."""
from pathlib import Path
import json
import math
base=Path(__file__).resolve().parents[1]
root=base/'paper gpt/dp-fl-ids code/results/boundary_extension_20261001'
success=0; failures=[]
for path in root.glob('*/record.json'):
    r=json.loads(path.read_text())
    if r['status']!='completed':
        failures.append({'name':r['config']['name'],'status':r['status']})
        continue
    assert r['completed_round_count']==30
    assert [e['round'] for e in r['completed_rounds']]==list(range(1,31))
    for e in r['completed_rounds']:
        a=e['attack_measurement']
        assert len(a['clients'])==3
        assert {c['client_id'] for c in a['clients']}==set(e['byzantine_clients'])
        assert set(e['byz_selected'])==set(e['selected']) & set(e['byzantine_clients'])
        for c in a['clients']:
            assert math.isfinite(c['perturbation_l2']) and c['perturbation_l2']>=0
            for denominator,key in [(c['pre_attack_update_l2'],'perturbation_over_pre_attack_update'),
                                    (a['honest_rms_radius'],'perturbation_over_honest_radius')]:
                if denominator:
                    assert math.isclose(c[key],c['perturbation_l2']/denominator,abs_tol=1e-12)
                else: assert c[key] is None
        if r['config']['attack']=='boundary_krum':
            search=a['search']; assert len(search['candidates'])==41
            feasible=[c for c in search['candidates'] if c['all_attackers_selected']]
            assert search['feasible_candidate_found']==bool(feasible)
            if feasible:
                assert search['chosen_scale']==max(c['scale'] for c in feasible)
                assert set(e['byz_selected'])==set(e['byzantine_clients'])
            else:
                assert search['chosen_scale'] is None
                assert all(c['perturbation_l2']==0 for c in a['clients'])
    f=r['result']['final']; cm=f['confusion_matrix']
    assert sum(map(sum,cm))==f['evaluated_rows']==4673255
    assert math.isclose(sum(cm[i][i] for i in range(34))/4673255,f['accuracy'],abs_tol=1e-12)
    if r['config']['dp']:
        spent=max(r['result']['privacy_accounting']['epsilon_by_client'].values())
        assert spent<=r['config']['target_epsilon']+1e-9
    success+=1
print(json.dumps({'successful_records_checked':success,'failures':failures,
                  'remaining':36-success-len(failures)},indent=2))
