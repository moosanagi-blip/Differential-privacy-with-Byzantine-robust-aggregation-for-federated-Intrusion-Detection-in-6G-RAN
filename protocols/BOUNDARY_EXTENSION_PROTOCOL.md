# Targeted reviewer-gap extension

Existing 102-condition results remain unchanged. New code adds a separate
boundary_krum attack and same-round pre-attack-to-submitted displacement
measurements for both the original proxy and the new attack.

The new attack enumerates 41 scales from zero to the declared cap on the
same honest-mean ray, evaluates the actual implemented Multi-Krum selector
with every colluding payload present, and retains the largest tested scale
for which all malicious clients are selected. Ties use the actual selector.
If no candidate qualifies, it submits unmodified local payloads and records
that fallback. This is finite-grid boundary-aware search, not a continuous
or global optimum. All candidate selections are retained.

Targeted proposed matrix: two attacks (proxy and boundary-aware), two privacy
settings (off and epsilon 5), caps 0.5/2/5, and paired partition/training
seeds 42/43/44: 36 runs. Other settings match the corrected campaign; only
static three-client coalitions and Multi-Krum with assumed f=3 are used.
Old proxy runs lack pre-attack payload measurements, so cannot supply the
new displacement evidence without instrumented reruns. No historical run
is overwritten. No tuning based on held-out utility.

Measure L2 submitted-minus-pre-attack local payload, divide by that client's
pre-attack increment norm and by the honest RMS radius where nonzero.
Undefined ratios are null. Under DP, the pre-attack payload already contains
training noise: this does not measure displacement from a non-private
counterfactual. Save each attacker separately with global client identity.

Gate: unit/regression tests, then a one-round full-shard timing and integration
check at the ordinary 30-slice size before freezing launch/runtime estimates.
No extension training has been launched by this protocol document.
