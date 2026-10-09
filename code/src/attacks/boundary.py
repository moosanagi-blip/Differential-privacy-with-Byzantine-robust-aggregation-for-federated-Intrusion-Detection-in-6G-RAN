"""Finite-grid attack using the implemented selector; not a global optimum."""
import numpy as np
from ..fl.params import flatten
from ..fl.aggregation import multi_krum
from .byzantine import _replace_flat

def craft_boundary(updates, global_params, positions, f_est, max_scale=2.):
    if not np.isfinite(max_scale) or max_scale < 0:
        raise ValueError('Finite nonnegative search cap required')
    positions = list(positions)
    if len(set(positions)) != len(positions) or any(i < 0 or i >= len(updates) for i in positions):
        raise ValueError('Invalid malicious positions')
    honest = [u for i,u in enumerate(updates) if i not in positions]
    if not positions or not honest:
        raise ValueError('Both malicious and honest clients required')
    g = flatten(global_params)
    direction = np.mean([flatten(u)-g for u in honest],axis=0)
    chosen = updates
    scale_chosen = None
    trace = []
    for scale in np.linspace(0.,max_scale,41):
        trial = _replace_flat(updates,global_params,positions,g+(1-scale)*direction)
        _,selected,scores = multi_krum(trial,f_est)
        feasible = set(positions).issubset(selected)
        trace.append({'scale':float(scale),'all_attackers_selected':feasible,
                      'selected':[int(i) for i in selected],
                      'malicious_scores':[float(scores[i]) for i in positions]})
        if feasible:
            chosen = trial
            scale_chosen = float(scale)
    return chosen, {'search':'41-point ray grid; largest scale with all attackers selected',
                    'chosen_scale':scale_chosen,'feasible_candidate_found':scale_chosen is not None,
                    'fallback':'unmodified pre-attack updates if no feasible candidate',
                    'candidates':trace}

def perturbation_metrics(before,after,global_params,positions):
    g=flatten(global_params)
    honest=np.stack([flatten(u) for i,u in enumerate(before) if i not in positions])
    centroid=honest.mean(axis=0)
    radius=float(np.sqrt(np.mean(np.sum((honest-centroid)**2,axis=1))))
    clients=[]
    for i in positions:
        original=flatten(before[i]); submitted=flatten(after[i])
        distance=float(np.linalg.norm(submitted-original))
        baseline=float(np.linalg.norm(original-g))
        clients.append({'position':int(i),'perturbation_l2':distance,
            'pre_attack_update_l2':baseline,
            'perturbation_over_pre_attack_update':distance/baseline if baseline else None,
            'perturbation_over_honest_radius':distance/radius if radius else None})
    return {'reference':'same-round pre-attack local payload; includes DP noise when enabled',
            'honest_rms_radius':radius,'clients':clients}
