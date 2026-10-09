"""Every word the demo says.

Kept apart from the logic so the explanations can be reworded — for a different
audience, a shorter meeting, a different supervisor — without touching anything
that computes a number.

Each step has a ``title``, a ``what`` (plain English: what this step is about to
do), a ``why`` (why it exists / which Research Objective it serves), and usually
a ``takeaway`` written after the numbers are in. Takeaways that need live values
are formatted by the step function instead; the ones here are static.

House style: no jargon without an immediate gloss, second person avoided, and
every claim either visible on screen or traceable to a file in ``results/``.
"""

from __future__ import annotations

TITLE = "DP-FL-IDS  |  COMPLETE NARRATED WALKTHROUGH OF THE IMPLEMENTATION"

_INTRO_LIVE = (
    """This program runs the complete thesis framework — Privacy-Preserving
    Federated Learning for Intrusion Detection in 6G Radio Access Networks — and
    explains every step as it happens. Nothing here is pre-recorded or mocked
    up: every accuracy, every epsilon, every exclusion rate printed below is
    computed live by the same code that produced the thesis results."""
)

_INTRO_REPLAY = (
    """This program walks through the complete thesis framework —
    Privacy-Preserving Federated Learning for Intrusion Detection in 6G Radio
    Access Networks — explaining every step. It is running in replay mode, so
    the figures below are read back from the experiment campaign's own result
    files rather than recomputed; each one names its source on screen."""
)

_INTRO_REST = (
    """The problem in one sentence: 6G base stations (gNBs) need an intrusion
    detection system, but the traffic they would have to pool in order to train
    one is exactly the data that must not be pooled. The framework answers that
    with three mechanisms working together — Federated Learning so raw traffic
    never leaves a base station, Differential Privacy so the model updates that
    do travel carry a mathematically bounded amount of information, and
    Multi-Krum aggregation so a minority of malicious base stations cannot
    poison the shared model.""",

    """The walkthrough follows the pipeline in order: first the data, then the
    model, then one federated round dissected in slow motion, and then the four
    Research Objectives in turn. Steps that train something print their live
    result beside the reported figure for the same configuration, so the gap
    between this shorter run and the 30-round campaign is visible rather than
    hidden.""",
)


def intro(live: bool = True) -> tuple[str, ...]:
    """Opening paragraphs. The first differs by mode: a replay run must not
    claim its numbers were computed live."""
    return ((_INTRO_LIVE if live else _INTRO_REPLAY),) + _INTRO_REST

def _duration(seconds: float) -> str:
    """'about 40 seconds' / 'about 3 minutes' / 'about 1.5 hours'."""
    if seconds < 90:
        return f"about {seconds:.0f} seconds"
    if seconds < 5400:
        return f"about {seconds / 60:.0f} minutes"
    return f"about {seconds / 3600:.1f} hours"


def runtime_explainer(model: str, rounds: int, n_runs: int = 20,
                      dataset: str = "data_full", n_train: int = 0,
                      n_test: int = 0, chunked: bool = False,
                      sec_plain: float = 0.0, sec_dp: float = 0.0
                      ) -> tuple[str, ...]:
    """Answer the obvious question before anyone has to ask it: what is about to
    happen, and how long will it take?

    Every figure here is derived from the dataset actually in use and from this
    project's own measured round times, so the estimate cannot drift away from
    what the run then does.
    """
    total = n_train + n_test
    per_gnb = n_train // max(rounds, 1) // 10

    return (
        """BEFORE ANYTHING RUNS — a word on what is about to happen and how long
        it takes. The framework here is complete: nothing is mocked, no number is
        hard-coded, and every mechanism really executes.""",

        f"""This run is on the COMPLETE CICIoT2023 corpus: {total:,} flow records
        — {n_train:,} distributed across the 10 gNBs and {n_test:,} held out for
        testing. Nothing is subsampled, and no part of the corpus is set
        aside.""",

        f"""That is why it is slow. Each of the {rounds} rounds trains every gNB
        on a disjoint slice of its own shard, so the {rounds} rounds together
        consume all {n_train:,} training records exactly once — no row is used
        twice and none is skipped. A round is roughly {per_gnb:,} records per
        gNB, which on the {model.upper()} takes {_duration(sec_plain)} without
        differential privacy and {_duration(sec_dp)} with it.""",

        f"""About {n_runs} separate federated experiments are run below. Where
        two steps need the identical configuration, it is computed once and
        reused — the results are seeded and deterministic, so recomputing it
        would change nothing except how long you wait. Differentially private
        rounds are the slow ones, roughly three to four times the cost of an
        ordinary round, because Opacus computes a separate gradient for every
        individual sample in order to clip each one.""",

        """Every experimental step prints its live number beside the reported
        figure for the same configuration (30 rounds on this same corpus, read
        from results/Research Objective 1..4/), so the two are never confused. This run
        is shorter than the reported campaign, so judge the shape of the result
        here and the magnitude from the reported column.""",
    )

STEPS: dict[int, dict[str, object]] = {

    1: dict(
        title="Environment and reproducibility",
        ro=None,
        what="""Check that the three libraries the framework depends on are
            present and report their versions, the compute device, and the
            random seed. PyTorch does the training, Opacus provides the
            differentially private optimiser and the privacy accountant, and
            Flower provides the federated learning framework.""",
        why="""Every number in this run is seeded, so the same command on the
            same machine reproduces the same results. Recording the versions
            here is what makes that claim checkable rather than decorative.""",
    ),

    2: dict(
        title="The raw dataset and preprocessing",
        ro=None,
        what="""Take real CICIoT2023 network flow records off disk and run them
            through the actual preprocessing pipeline: drop malformed rows,
            coerce numeric columns, encode the attack labels, and standardise
            the features. To keep this step short it processes one raw file
            capped at a few rows per class — the same function, a smaller
            slice.""",
        why="""CICIoT2023 is a published IoT intrusion dataset of about 46.7
            million flow records carrying 34 distinct attack labels grouped into
            8 categories, described by 46 network features. It is the empirical
            grounding for the whole thesis, so the demo starts by showing the
            raw material rather than asserting it.""",
    ),

    3: dict(
        title="6G RAN feature augmentation",
        ro=None,
        what="""Append the seven synthetic radio-access features that turn a
            generic IoT flow record into a 6G RAN observation: RSRP, SINR, RSRQ,
            PRB utilisation, handover rate, beam-failure count and throughput.
            Their sampling ranges follow 3GPP TS 38.215 / 38.214 / 38.331, and
            they are conditioned on the attack category so that, for example, a
            flooding attack shows the PRB exhaustion it would really cause.""",
        why="""No public dataset carries both real 6G radio telemetry and
            labelled intrusions, so the RAN layer has to be synthesised. This is
            a stated limitation of the work, not a hidden assumption — the
            detection results depend on it and the thesis says so. 46 network
            features plus 7 RAN features gives the 53-dimensional vector every
            model in this framework consumes.""",
    ),

    4: dict(
        title="Splitting the data across 10 gNBs (non-IID)",
        ro=None,
        what="""Load the ten client shards and the held-out global test set, and
            show how unevenly the attack categories are spread across them. The
            split uses a Dirichlet distribution with alpha = 0.3, which
            concentrates each category on a few base stations instead of
            spreading it evenly.""",
        why="""Real base stations do not see identical traffic — a tower in an
            industrial estate sees different attacks from one in a residential
            area. That imbalance, called non-IID data, is the single thing that
            makes federated learning hard, so building it in deliberately is
            what makes the evaluation honest. The global test set is never
            trained on; it is the neutral ground the shared model is judged on.""",
    ),

    5: dict(
        title="The intrusion-detection model",
        ro=None,
        what="""Build the classifiers and count their parameters. Two
            architectures matter here: an MLP, which carries the quantitative
            results because it trains fast and tolerates DP noise well, and an
            LSTM built on Opacus's DPLSTM, which is the thesis's primary
            architecture for sequential traffic.""",
        why="""The model has to run identically with and without differential
            privacy, otherwise the privacy comparison would be confounded by an
            architecture change. Opacus's DPLSTM exists precisely because a
            standard PyTorch LSTM cannot expose per-sample gradients, which
            DP-SGD requires.""",
    ),

    6: dict(
        title="One federated round, dissected",
        ro=1,
        what="""Run a single federated round with every internal action printed:
            the server broadcasting the global weights, each of the ten gNBs
            training locally on data that never moves, each returning only its
            updated weights, the server aggregating them, and the new global
            model being scored on the held-out test set.""",
        why="""Everything that follows is this loop repeated. Watching it once
            in full makes the rest of the run readable — from here on the same
            process is summarised to one line per round.""",
        takeaway="""That single loop is the entire mechanism. Ten base stations
            improved one shared model, and the only thing that crossed the
            network was a list of weights.""",
    ),

    7: dict(
        title="Baseline — plain federated learning",
        ro=1,
        what="""Train the shared model across all ten gNBs with straightforward
            FedAvg aggregation: no differential privacy, no attackers. Each
            round the server takes a weighted average of the ten updates, in
            proportion to how much data each client holds.""",
        why="""This is the reference point. Privacy costs accuracy and
            robustness costs accuracy, and neither cost means anything without a
            baseline to measure it against. This number is the ceiling the rest
            of the run is traded against.""",
    ),

    8: dict(
        title="Adding (epsilon, delta)-differential privacy",
        ro=1,
        what="""Repeat the same training with DP-SGD. Two things now happen to
            every batch: each individual sample's gradient is clipped to a fixed
            norm C = 1.0 so no single flow record can dominate the update, and
            Gaussian noise of scale sigma x C is added to the sum. A Renyi
            differential privacy accountant tracks the cumulative privacy loss,
            and the framework calibrates sigma by bisection so the total budget
            is spent exactly by the final round.""",
        why="""Federated learning alone is not private. Model updates leak:
            gradient-inversion attacks can reconstruct training records from
            them, and membership-inference attacks can tell whether a particular
            record was in the training set. Differential privacy replaces the
            hope that updates are safe with a bound on how much they can
            reveal.""",
    ),

    9: dict(
        title="Framework ablation — which component contributes what",
        ro=1,
        what="""Run the four combinations that make up the integrated framework:
            federated learning alone, plus differential privacy, plus Multi-Krum
            aggregation, and finally both together. Presented as a table so each
            component's cost is separable.""",
        why="""RO1 asks for an integrated framework, and integration is a claim
            that has to be tested rather than asserted. The interesting cell is
            the last one: if DP and Multi-Krum interfered with each other, the
            combination would be worse than either alone.""",
    ),

    10: dict(
        title="Privacy-utility characterisation",
        ro=3,
        what="""Sweep the privacy budget across a range of epsilon values and
            record accuracy at each. Smaller epsilon means a stronger privacy
            guarantee, which means more noise, which means lower accuracy — this
            step measures the shape of that curve rather than assuming it.""",
        why="""RO3 asks what privacy actually costs in this setting. An operator
            deploying the system has to choose an epsilon to satisfy a
            regulator; that choice is only informed if someone has mapped the
            trade-off. A smooth, monotonic curve means the dial behaves
            predictably.""",
    ),

    11: dict(
        title="Adaptive privacy-budget scheduling",
        ro=2,
        what="""Spend the same total privacy budget unevenly across rounds and
            see whether it helps. Three schedules are compared at a matched
            total epsilon: static (constant noise), decreasing noise (loud early,
            quiet late), and increasing noise (quiet early, loud late). The
            framework calibrates any schedule shape to the same total budget via
            RDP composition, so the comparison is budget-fair by construction.
            The step then runs the optimiser ablation that turned out to matter
            more: Adam against SGD at the same epsilon.""",
        why="""RO2 proposed adaptive scheduling as the central novelty. The
            honest result is that it does not beat a uniform schedule — under
            RDP composition at a fixed budget, constant noise is near-optimal,
            and a bounded sweep of six shapes confirmed it. The real
            privacy-utility improvement came from somewhere else: switching the
            DP runs from Adam to plain SGD, because Adam's per-coordinate
            rescaling fights the clip-and-noise mechanism. Reporting a
            well-tested negative result alongside the mechanism that did work is
            the contribution.""",
    ),

    12: dict(
        title="Byzantine robustness",
        ro=4,
        what="""Turn a fraction of the gNBs malicious and compare aggregation
            rules. The attackers run a sign-flip attack: each round they invert
            their model update and scale it up, actively dragging the global
            model away from anything useful. FedAvg, Krum, Multi-Krum and
            trimmed-mean each face the same attack. Watch the byz_kept counter —
            it reports how many poisoned updates made it into the aggregate.""",
        why="""A base station can be compromised, and FedAvg averages whatever
            it is given, so a single determined attacker can destroy the shared
            model. Multi-Krum scores each update by its summed distance to its
            nearest neighbours: honest updates cluster, poisoned ones sit apart,
            and only the closest n-f are averaged. RO4 asks whether that defence
            holds in this setting.""",
    ),

    13: dict(
        title="Does DP noise blind the defence?",
        ro=4,
        what="""The question RO4 exists to answer. Multi-Krum identifies
            attackers by how far their updates sit from everyone else's;
            differential privacy deliberately adds noise to every update. If
            that noise swamps the distance signal, the two mechanisms cannot be
            deployed together. Measured two ways: the Byzantine exclusion rate
            with and without DP under a matched attack, and a micro-benchmark
            that injects increasing noise into a fixed pool of updates and
            tracks the honest-versus-Byzantine score gap directly.""",
        why="""This is the composition question, and a negative answer would
            undermine the whole framework — there would be no point offering
            privacy and robustness together if enabling one disabled the
            other.""",
    ),

    14: dict(
        title="Engine parity — the genuine Flower implementation",
        ro=1,
        what="""Re-run one configuration through the real Flower federated
            learning framework — actual NumPyClient objects and an actual
            Strategy subclass exchanging FitIns and FitRes messages — and check
            it lands on the same numbers as the sequential reference engine.""",
        why="""The sequential engine is deterministic and easy to reason about,
            which is why it produced the results. But a thesis claiming a
            federated system should demonstrate it on a real federated
            framework, and the two agreeing is what shows the sequential engine
            is a faithful reference rather than a simplification.""",
    ),

    15: dict(
        title="Where the detector succeeds and where it fails",
        ro=3,
        what="""Break the final model's performance down per attack class
            instead of reporting a single accuracy, and show the gap between
            overall accuracy and macro-averaged F1.""",
        why="""Accuracy alone flatters this model. High-volume floods —
            DDoS, DoS, Mirai — are detected almost perfectly, while rare classes
            with only a handful of samples each are missed almost entirely. That
            is why macro-F1 sits well below accuracy: it weights every class
            equally regardless of size. This is a class-imbalance effect
            inherited from the corpus itself — the largest attack family
            outnumbers the smallest by roughly 6000 to 1 — it is expected, and
            the thesis reports it as a limitation rather than burying it.""",
    ),

    16: dict(
        title="Threat model",
        ro=1,
        what="""Summarise the documented threat model: what is being protected,
            which adversaries are in scope, what is assumed trustworthy, and
            which threats are explicitly out of scope.""",
        why="""RO1 requires a formal threat model, and its value is as much in
            what it excludes as what it covers. Naming the out-of-scope attacks
            is what stops the framework's guarantees from being read more
            broadly than they were tested.""",
    ),

    17: dict(
        title="Summary",
        ro=None,
        what="""Collect every experiment run in this session into one table, map
            each Research Objective to the evidence produced for it, and point
            at the files where the full-scale results live.""",
        why="""So that what was just demonstrated can be checked afterwards
            rather than taken on trust.""",
    ),
}


def step_meta(n: int) -> dict:
    return STEPS.get(n, {"title": f"Step {n}", "what": "", "why": "", "ro": None})


# ── closing text ───────────────────────────────────────────────────────────
_CLOSING_BODY = (
    """RO1 — the integrated FL + DP + Multi-Krum framework, built up component by
    component and ablated (steps 6 to 9). RO2 — adaptive epsilon schedules
    calibrated to a fixed budget, plus the optimiser ablation that produced the
    actual improvement (step 11). RO3 — the privacy-utility curve and the
    per-class breakdown (steps 10 and 15). RO4 — Multi-Krum's robustness under
    attack and its clean composition with DP noise (steps 12 and 13)."""
)


def closing(live: bool = True) -> tuple[str, ...]:
    lead = ("All four Research Objectives were exercised live."
            if live else
            "All four Research Objectives are covered, from saved results.")
    return (lead + " " + _CLOSING_BODY,)
