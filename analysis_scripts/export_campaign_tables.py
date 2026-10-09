"""Generate highlighted LaTeX tables from the audited summary, preserving failures."""
from pathlib import Path
import json
base=Path(__file__).resolve().parents[1]
source=base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26/analysis_initial/summary.json'
groups=list(json.loads(source.read_text()).items())
def label(name):
    return (name.replace('primary_','').replace('nonprivate','No DP')
        .replace('dp5','DP5').replace('multikrum','MK').replace('uniformavg','Equal mean')
        .replace('fedavg','FedAvg').replace('radio_masked','Radio masked')
        .replace('assumed_f','Assumed f=').replace('epsilon','Epsilon ')
        .replace('cap','Cap ').replace('_',' / '))
parts=[]
for part in range(2):
    lines=[r'\begin{table}[p]',r'\centering',r'\begingroup\color{blue}\scriptsize',
        r'\caption{Corrected campaign summary, part '+str(part+1)+r'. DP5 denotes target $\epsilon=5$; MK denotes Multi-Krum. Values are mean $\pm$ sample SD across three paired seeds. A dash means an incomplete triplet, not zero utility.}',
        r'\label{tab:campaign-summary-'+str(part+1)+'}',
        r'\begin{tabular}{@{}lcrrr@{}}',r'\toprule',
        r'Condition & Success & Accuracy (\%) & Macro-F1 & FPR (\%) \\',r'\midrule']
    for name,g in groups[part*17:(part+1)*17]:
        n=sum(r['status']=='completed' for r in g['replicates'])
        a=g['aggregate']; values=[]
        for metric,factor in [('accuracy',100),('macro_f1',1),('fpr',100)]:
            values.append(f"${factor*a[metric]['mean']:.3f}\\pm{factor*a[metric]['sample_sd']:.3f}$" if a else '---')
        lines.append(label(name)+f' & {n}/3 & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table}']
    parts.extend(lines)
destination=base/'paper gpt/paper/generated/corrected_campaign_tables.tex'
destination.write_text('\n'.join(parts)+'\n',encoding='utf-8')
print(f'Generated two tables containing {len(groups)} conditions.')
