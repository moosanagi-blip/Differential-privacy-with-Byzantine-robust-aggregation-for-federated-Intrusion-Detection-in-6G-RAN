from pathlib import Path
import json, os, shutil
base=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(base/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=json.loads((base/'output/final_gap_statistics_20261006/statistics.json').read_text())
out=base/'output/final_gap_statistics_20261006'
colors=['#0072B2','#D55E00','#009E73']
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    shutil.copy2(out/f'{name}.png',base/f'paper gpt/paper/figures/{name}.png')
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for ax,key,title in zip(axes,['perturbation_l2','perturbation_over_honest_radius'],['Absolute payload displacement','Displacement / honest RMS radius']):
    for i,seed in enumerate((42,43,44)):
        rows=[r for r in d['displacement_seeds'] if r['seed']==seed]
        ax.plot([r['epsilon'] for r in rows],[r[key] for r in rows],marker=['o','s','^'][i],color=colors[i],label=f'Seed {seed}')
    ax.set_xticks([1,5,10]);ax.set_xlabel('Target epsilon');ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
axes[0].set_ylabel('Within-run median');axes[0].legend(fontsize=8)
save(fig,'reviewer_final_displacement')
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for i,row in enumerate(d['noise_seeds']):
    axes[0].scatter(i,row['mean_round_rank_rejection_correlation'],color=colors[i],marker=['o','s','^'][i],s=45)
    axes[1].bar(i-.18,row['rescue_fraction_of_90_rejections']*100,.36,color='#0072B2',label='Rejected → selected (90 cases/seed)' if i==0 else None)
    axes[1].bar(i+.18,row['harm_fraction_of_210_selections']*100,.36,color='#D55E00',label='Selected → rejected (210 cases/seed)' if i==0 else None)
    axes[1].text(i+.18,2,'0%',ha='center',fontsize=8)
for ax in axes:ax.set_xticks(range(3),['42','43','44']);ax.set_xlabel('Seed');ax.grid(axis='y',alpha=.2)
axes[0].set_ylim(0,1);axes[0].set_ylabel('Mean round-wise rank correlation');axes[0].set_title('Payload-change magnitude vs rejection',fontsize=10)
axes[1].set_ylim(0,125);axes[1].set_ylabel('Conditional selection-flip rate (%)');axes[1].set_title('Suppress only the focal client’s noise',fontsize=10);axes[1].legend(fontsize=7,loc='upper center')
save(fig,'reviewer_final_noise_selection')
print('Exported two PNG/SVG figures and manuscript PNG copies.')
