# Final reviewer measurements — 3 October 2026

Nine new full-corpus conditions: static proxy attack, cap 2, epsilon 1 and
10, seeds 42/43/44 (six); clean Multi-Krum epsilon 5, same seeds (three).
Use original 53-feature partitions, 30 disjoint round slices, established
MLP/SGD/float64 settings. Existing epsilon-5 displacement is the matched
reference. No frozen code or existing results are changed.

Clean tests replay every local update from the same global parameters and
seed, retaining clipping, Poisson sampling, momentum and dropout. Replay
consumes the usual Gaussian draws but suppresses their gradient contribution.
Restore actual-run RNG state after replay. This is a noise-suppressed clipped
DP-SGD counterfactual, NOT ordinary SGD. Replayed gradient trajectories may
diverge; payload distance measures total local effect of noise, not a raw
Gaussian-vector norm. Replay payloads are never fed into live training.

Compare actual selection with (a) all replay payloads and (b) replacing only
one client's payload at a time, holding the nine other actual payloads fixed.
Save per-client distance, scores, and actual/counterfactual selection per round.
Report within-seed association and selection flip frequencies, then seed-level
summaries; clients/rounds are not independent replicates. The intervention
isolates a local replay effect conditional on the actual global trajectory,
not the long-term utility of a noise-free training trajectory.

Counterfactual diagnostics are non-private research artifacts; the live-run
epsilon bound does NOT protect their release. Do not automatically include
raw counterfactual diagnostics in a public upload bundle.

Run a one-round full-size clean integration test first; label it smoke-only.
Never count smoke results as final evidence. Preserve failures explicitly.
Do not update manuscript conclusions before full records have been audited.
