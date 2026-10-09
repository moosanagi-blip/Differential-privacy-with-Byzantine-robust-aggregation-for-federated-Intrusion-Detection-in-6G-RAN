"""Saved selection margins and submitted-payload geometry; no new training."""
from pathlib import Path
import hashlib
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1]
root=base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26'
summary=json.loads((root/'analysis_initial/summary.json').read_text())
out=base/'output/reviewer_figures_20260930'
plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
fig,axes=plt.subplots(3,2,figsize=(12,10))
colors=['#0072B2','#D55E00','#009E73']
saved=[]
for row,(name,epsilon) in enumerate([('epsilon1_static',1),('primary_dp5_multikrum_static',5),('epsilon10_static',10)]):
    for j,rep in enumerate(summary[name]['replicates']):
        assert rep['status']=='completed'
        source=Path(rep['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==rep['sha256']
        record=json.loads(source.read_text())
        rounds=[]; margins=[]; ratios=[]
        for event in record['completed_rounds']:
            d=event['diagnostics']; selection=d['score_selection']
            cutoff=selection['max_selected_score']
            malicious=[c for c in selection['clients'] if c['malicious']]
            assert len(malicious)==3
            # Preserve exact records and flags; median is only a visual summary.
            normalized=[c['signed_margin']/abs(cutoff) for c in malicious] if cutoff else []
            margin=statistics.median(normalized) if normalized else None
            rounds.append(event['round']); margins.append(margin)
            ratios.append(d['malicious_distance_over_honest_radius'])
            saved.append({'target_epsilon':epsilon,'seed':rep['seed'],'round':event['round'],
                'cutoff':cutoff,'malicious':malicious,'median_normalized_margin':margin,
                'honest_rms_radius':d['honest_rms_radius'],
                'malicious_rms_distance_to_honest_centroid':d['malicious_rms_distance_to_honest_centroid'],
                'ratio':d['malicious_distance_over_honest_radius']})
        axes[row,0].plot(rounds,margins,color=colors[j],label=f'Seed {rep["seed"]}',lw=1.3)
        axes[row,1].plot(rounds,ratios,color=colors[j],label=f'Seed {rep["seed"]}',lw=1.3)
    axes[row,0].axhline(0,color='black',ls='--',lw=.8)
    axes[row,0].set(title=f'Target ε={epsilon}: malicious score margin',ylabel='Median (cutoff − score) / |cutoff|')
    axes[row,1].set(title=f'Target ε={epsilon}: submitted-update geometry',ylabel='Malicious RMS distance / honest RMS radius')
    for ax in axes[row]: ax.set(xlabel='Round',xlim=(1,30)); ax.grid(alpha=.2)
axes[0,0].legend(frameon=False,ncol=3)
fig.suptitle('Static adaptive Multi-Krum: decision margins and geometry across privacy budgets',fontsize=14)
fig.tight_layout(rect=(0,.055,1,.955))
fig.text(.5,.014,'Positive margin is below the selected-score cutoff; ties require actual selection flags.\nGeometry is measured on submitted parameters, not achieved attack displacement.',ha='center',fontsize=9)
for ext in ('png','svg'): fig.savefig(out/f'selection_geometry.{ext}',dpi=300)
with (out/'selection_geometry_values.json').open('w') as handle: json.dump(saved,handle,indent=2,allow_nan=False)
print(f'Exported {len(saved)} per-round records across 9 completed runs.')
