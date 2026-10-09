"""Sequencing, timing, transcript capture and error containment.

Error containment matters more than it sounds: this runs in front of a
supervisor. A step that throws should say so clearly and let the demonstration
carry on, not dump a traceback and abort with nine steps unshown.
"""

from __future__ import annotations

import io
import re
import sys
import time
import traceback
from pathlib import Path

from ..data.config import PROJECT_ROOT
from ..utils import console
from . import narration
from .steps import ALL_STEPS, PROFILES, Ctx

OUTDIR = PROJECT_ROOT / "results" / "demo_run"

# Seconds of training per record, (non-DP, DP). A round's cost is proportional
# to the rows it touches, not to the number of rounds, so the estimate has to be
# per-record: chunked rounds divide one pass over the corpus between them, and
# halving `rounds` doubles the size of each one.
#
# Measured from this project's own 30-round full-corpus campaign
# (results/Research Objective */*.json, 1.24M records per round): 28 s non-DP
# and 115 s with DP. The LSTM entry is an estimate, not a measurement from this
# campaign. DP costs more because Opacus computes a per-sample gradient in
# order to clip each one.
_RECORD_SECONDS = {
    "mlp": (22.5e-6, 92.0e-6),
    "lstm": (9.1e-4, 2.9e-3),
    "transformer": (9.1e-4, 2.9e-3),
}


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Tee(io.TextIOBase):
    """Mirror stdout into a transcript file.

    The file copy is stripped of colour codes and of the in-place progress
    lines, so the transcript reads as plain text — it is meant to be opened
    afterwards, or handed over, not replayed in a terminal.
    """

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s: str) -> int:
        self.stream.write(s)
        if "\r" not in s:
            self.fh.write(_ANSI.sub("", s))
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()


def parse_steps(spec: str | None) -> list[int]:
    """``"7-13"`` / ``"1,4,12"`` / ``None`` -> ordered step numbers."""
    if not spec:
        return sorted(ALL_STEPS)
    picked: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            picked.update(range(int(a), int(b) + 1))
        else:
            picked.add(int(part))
    return sorted(n for n in picked if n in ALL_STEPS)


def _round_seconds(model: str, n_train: int, rounds: int) -> tuple[float, float]:
    """Seconds for one federated round, (non-DP, DP), on this dataset.

    Chunked rounds split one pass over the corpus between them, so a round costs
    ``n_train / rounds`` records regardless of how many rounds there are.
    """
    per_plain, per_dp = _RECORD_SECONDS.get(model, _RECORD_SECONDS["lstm"])
    rows = n_train / max(rounds, 1)
    return per_plain * rows, per_dp * rows


def _estimate_minutes(profile: dict, steps: list[int], model: str,
                      n_train: int) -> float:
    """Rough wall-clock estimate. Counts only runs the cache cannot serve."""
    r = profile["rounds"]
    s_plain, s_dp = _round_seconds(model, n_train, r)
    plain, dp = s_plain * r, s_dp * r
    cost = {
        2: 5,                                        # preprocessing a small slice
        6: s_plain,                                  # one round only
        7: plain,
        8: dp,
        9: plain + dp,                               # 2 of the 4 come from cache
        10: dp * max(0, len(profile["eps_grid"]) - 1),
        11: dp * (len(profile["schedules"]) + 1),    # static is cached
        12: plain * max(0, len(profile["aggregators"]) *
                        len(profile["byz_fracs"]) - 2),
        13: plain + dp + 10,                         # + score-separation bench
        14: plain,                                   # sequential half is cached
    }
    return sum(cost.get(s, 1) for s in steps) / 60.0


def _count_runs(profile: dict) -> int:
    """How many federated experiments the sequence asks for, before the cache
    collapses duplicates. Used only in the opening explanation."""
    return (1                                     # the dissected single round
            + 1 + 1                               # baseline, DP
            + 4                                   # ablation
            + len(profile["eps_grid"])            # privacy-utility sweep
            + 1 + len(profile["schedules"]) + 2   # schedules + optimiser pair
            + len(profile["aggregators"]) * len(profile["byz_fracs"])
            + 2                                   # interaction
            + 2)                                  # engine parity


def _pause(enabled: bool, next_step: int | None = None) -> None:
    """Wait for Enter, naming what comes next so the presenter can see it."""
    if not enabled:
        return
    if next_step is None:
        console.prompt("--- press Enter to finish ---")
        return
    title = narration.step_meta(next_step)["title"]
    console.prompt(f"--- press Enter for STEP {next_step}: {title} ---")


def run_demo(args) -> int:
    profile = PROFILES[args.profile]
    steps = parse_steps(args.steps)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    transcript = OUTDIR / "transcript.txt"
    fh = transcript.open("w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, fh)

    lr = args.lr if args.lr is not None else (0.1 if args.optimizer == "sgd" else 1e-3)
    data_root = getattr(args, "data_root", None)
    base = dict(model=args.model, rounds=profile["rounds"], local_epochs=1,
                batch_size=64, num_clients=10, device=args.device,
                seed=args.seed, engine=args.engine, optimizer=args.optimizer,
                lr=lr, hidden=64, data_root=data_root,
                # A full-corpus shard cannot be trained on whole in one round;
                # chunk it so the rounds together make exactly one pass.
                round_chunks=bool(data_root))
    ctx = Ctx(args=args, profile=profile, outdir=OUTDIR, base=base)

    t_start = time.time()
    failures: list[tuple[int, str]] = []

    try:
        # ── opening ──────────────────────────────────────────────────────
        console.banner(narration.TITLE, color="magenta")
        console.say(*narration.intro(live=True))

        # Read the real shape of whichever dataset root is in use, so neither the
        # estimate nor the opening explanation can describe a dataset we are not
        # using.
        n_train = n_test = 0
        try:
            from ..data.loaders import DataRoot
            _r = DataRoot(data_root)
            man = _r.manifest()
            n_train = int(man.get("n_train") or sum(
                _r.client_rows(c) for c in range(_r.n_clients())))
            n_test = int(man.get("n_test") or 0)
        except Exception as exc:
            console.warn(f"Could not read the dataset manifest ({exc!r}) — the "
                         "opening estimate will be omitted.")

        est = _estimate_minutes(profile, steps, args.model, n_train)
        secs = est * 60
        if secs < 90:
            est_txt = f"about {secs:.0f} second{'' if round(secs) == 1 else 's'}"
        elif secs < 5400:
            est_txt = f"roughly {est:.0f} minutes"
        else:
            est_txt = f"roughly {est / 60:.1f} hours"
        n_txt = f"{len(steps)} step{'' if len(steps) == 1 else 's'}"
        console.say(f"""Settings for this run: model {args.model.upper()},
            {profile['rounds']} federated rounds per experiment, optimiser
            {args.optimizer.upper()}, device {args.device}, seed {args.seed},
            profile '{args.profile}'. {n_txt} to come, {est_txt} of computation
            on an idle machine — proportionally longer if something else is
            using the CPU.""")

        s_plain, s_dp = _round_seconds(args.model, n_train, profile["rounds"])
        console.rule("yellow")
        console.say(*narration.runtime_explainer(
            args.model, profile["rounds"], n_runs=_count_runs(profile),
            dataset=data_root, n_train=n_train, n_test=n_test,
            chunked=bool(base["round_chunks"]),
            sec_plain=s_plain, sec_dp=s_dp))
        console.rule("yellow")

        console.table(["step", "what it covers", "RO"],
                      [[str(n), str(narration.step_meta(n)["title"]),
                        f"RO{narration.step_meta(n)['ro']}"
                        if narration.step_meta(n)["ro"] else "-"]
                       for n in steps], aligns="<<<")

        if args.pause and not console.stdin_is_interactive():
            console.warn("Step-by-step pausing is on, but stdin is not a "
                         "terminal — the run will proceed without waiting.")
        elif args.pause:
            console.info("Pausing between steps. Press Enter to advance; "
                         "Ctrl-C to stop. Use --no-pause to run straight "
                         "through.")
        _pause(args.pause, steps[0] if steps else None)

        # ── steps ────────────────────────────────────────────────────────
        for i, n in enumerate(steps):
            meta = narration.step_meta(n)
            # Always number against the full sequence, so "STEP 12 of 17" means
            # the same thing whether or not --steps selected a subset.
            title = f"STEP {n} of {max(ALL_STEPS)}  —  {meta['title']}"
            if meta["ro"]:
                console.ro(int(meta["ro"]), title)
            else:
                console.banner(title)

            if meta.get("what"):
                console.say("WHAT THIS STEP DOES.  " + str(meta["what"]))
            if meta.get("why"):
                console.say("WHY IT MATTERS.  " + str(meta["why"]))

            t0 = time.time()
            try:
                ALL_STEPS[n](ctx)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failures.append((n, repr(exc)))
                console.warn(f"Step {n} failed: {exc!r}")
                console.warn("Continuing with the next step; the traceback is in "
                             f"{transcript.name}.")
                traceback.print_exc(file=fh)
            console.info(f"step {n} finished in {time.time() - t0:.1f}s")
            _pause(args.pause, steps[i + 1] if i + 1 < len(steps) else None)

        # ── closing ──────────────────────────────────────────────────────
        mins = (time.time() - t_start) / 60
        if failures:
            console.banner("RUN COMPLETE — with failures", color="yellow")
            for n, err in failures:
                console.warn(f"step {n}: {err}")
        else:
            console.banner("RUN COMPLETE", color="green")
        console.kv({"total wall-clock": f"{mins:.1f} minutes",
                    "transcript": str(transcript)})
        return 1 if failures else 0

    except KeyboardInterrupt:
        console.warn("\nInterrupted. Partial artefacts are in " + str(OUTDIR))
        return 130
    finally:
        sys.stdout = real_stdout
        fh.close()
