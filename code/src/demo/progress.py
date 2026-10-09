"""Turn engine ``on_event`` callbacks into live console output.

Without this the demo goes quiet for up to 50 s at a time: ``run_federated``
only prints once a round has finished, and a DP round spends that whole time
inside per-sample gradient clipping. An audience watching a blank terminal
assumes the thing has hung.

Two verbosity levels:

``"full"``   every gNB's local training is printed on its own line, plus the
             aggregation decision. Used by the step that dissects a single
             federated round, where the detail *is* the lesson.
``"round"``  one line per completed round, with an in-round counter that
             rewrites itself on a TTY so progress is visible without scrolling
             hundreds of lines past the viewer.
"""

from __future__ import annotations

import sys

from ..utils import console


class ProgressReporter:
    """Callable suitable for ``run_federated(on_event=...)``."""

    def __init__(self, level: str = "round", live_line: bool | None = None):
        self.level = level
        # Live rewriting needs a real terminal. Decide from the *original*
        # stdout, because the runner tees stdout to the transcript file, which
        # would otherwise report itself as not-a-tty and disable the counter.
        if live_line is None:
            live_line = bool(getattr(sys.__stdout__, "isatty", lambda: False)())
        self.live_line = live_line and level == "round"
        self._f_est = 0
        self._rounds = 0
        self._round = 0
        self._line_open = False

    # ── internals ────────────────────────────────────────────────────────
    def _clear_line(self) -> None:
        if self._line_open:
            sys.__stdout__.write("\r" + " " * 78 + "\r")
            sys.__stdout__.flush()
            self._line_open = False

    def _live(self, text: str) -> None:
        if not self.live_line:
            return
        sys.__stdout__.write("\r" + text[:78].ljust(78))
        sys.__stdout__.flush()
        self._line_open = True

    # ── the callback ─────────────────────────────────────────────────────
    def __call__(self, event: str, p: dict) -> None:
        fn = getattr(self, f"_on_{event}", None)
        if fn is not None:
            fn(p)

    def _on_run_start(self, p: dict) -> None:
        self._f_est = p.get("f_est", 0)
        self._rounds = p.get("rounds", 0)
        if self.level != "full":
            return
        console.kv({
            "clients (gNBs)": p["n_clients"],
            "aggregator": p["aggregator"],
            "differential privacy": "on" if p["dp"] else "off",
            "Byzantine clients": p["byzantine"] or "none",
        }, indent="    ")

    def _on_round_start(self, p: dict) -> None:
        self._round = p["round"]
        self._rounds = p["rounds"]
        if self.level == "full":
            sigma = p["sigma"]
            extra = f"   noise sigma = {sigma:.3f}" if sigma else ""
            print()
            console.info(f"round {p['round']}/{p['rounds']}: broadcasting global "
                         f"weights to every gNB{extra}")

    def _on_client_start(self, p: dict) -> None:
        if self.level == "full":
            return
        self._live(f"      round {self._round}/{self._rounds}  training gNB "
                   f"{p['index']}/{p['total']} ({p['n_samples']} samples) ...")

    def _on_client_end(self, p: dict) -> None:
        if self.level != "full":
            return
        tag = "  <- BYZANTINE, update poisoned" if p["is_byzantine"] else ""
        dp = f"  DP sigma={p['noise_multiplier']:.2f}" if p.get("noise_multiplier") else ""
        console.substep(f"gNB {p['client']:02d}  trained on {p['n_examples']:>5} "
                        f"local samples  ({p['seconds']:.1f}s){dp}{tag}")

    def _on_aggregate_start(self, p: dict) -> None:
        if self.level == "full":
            console.info(f"all {p['n_updates']} updates returned; server now "
                         f"aggregates with {p['aggregator']}")

    def _on_aggregate_end(self, p: dict) -> None:
        if self.level != "full":
            return
        kept, excl = p["selected"], p["excluded"]
        if p["aggregator"] in ("krum", "multikrum", "trimmedmean"):
            console.substep(f"kept {len(kept)} updates {kept}")
            console.substep(f"discarded {len(excl)} as outliers {excl}")
            if p["f_est"]:
                slipped = len(p["byz_selected"])
                verdict = ("every Byzantine update was excluded" if slipped == 0
                           else f"{slipped} Byzantine update(s) slipped through")
                console.substep(f"-> {verdict}")
        else:
            console.substep(f"averaged all {len(kept)} updates (no outlier filtering)")

    def _on_round_end(self, p: dict) -> None:
        self._clear_line()
        eps = f"  eps={p['cum_epsilon']:.2f}" if p["cum_epsilon"] else ""
        byz = (f"  byz_kept={len(p['byz_selected'])}/{self._f_est}"
               if self._f_est else "")
        line = (f"    round {p['round']:>2}/{p['rounds']}  acc={p['accuracy']:.3f}"
                f"  macro-F1={p['macro_f1']:.3f}{eps}{byz}  ({p['time_s']:.1f}s)")
        print(line)

    def _on_run_end(self, p: dict) -> None:
        self._clear_line()


def reporter(level: str = "round") -> ProgressReporter:
    return ProgressReporter(level=level)
