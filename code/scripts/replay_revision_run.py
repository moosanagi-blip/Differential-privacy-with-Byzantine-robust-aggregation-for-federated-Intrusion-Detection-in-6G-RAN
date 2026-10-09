"""Failure-aware replay of one saved revision configuration, unchanged.

Outputs go to a separate directory. A numerical failure has no final accuracy.
Example: python scripts/replay_revision_run.py adaptive_static__seed44
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from src.experiment_config import RunConfig
from src.fl.simulation import run_federated

OUT = CODE.parent / "results" / "revision_full_corpus"


def execute(config, runner=run_federated):
    """Keep completed rounds and failure context without assigning fake metrics."""
    context, completed = {}, []
    provenance = {}
    current_round = None

    def observe(event, payload):
        nonlocal current_round
        if event == "provenance":
            provenance.update(payload)
            return
        if "round" in payload:
            current_round = payload["round"]
        context.clear()
        context.update(event=event, round=current_round)
        context.update(payload)
        if event == "round_end":
            completed.append(dict(payload))
            print(f"Completed round {payload['round']}: loss={payload['loss']:.6g}", flush=True)

    try:
        result = runner(config, verbose=False, on_event=observe)
    except FloatingPointError as exc:
        return dict(status="numerical_failure", error=str(exc),
                    provenance=provenance,
                    failure_context=context.copy(), completed_rounds=completed,
                    completed_round_count=len(completed), final=None)
    return dict(status="completed", result=result, provenance=provenance,
                completed_round_count=len(completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Saved run filename stem")
    args = parser.parse_args()
    if Path(args.run).name != args.run or any(c in args.run for c in ('/', '\\', ':')):
        parser.error("Use a filename stem, not a path")
    source = OUT / "runs" / (args.run + ".json")
    saved = json.loads(source.read_text(encoding="utf-8"))
    config = RunConfig.from_dict(saved["config"])
    replay_dir = OUT / "failure_aware_replays"
    replay_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = replay_dir / f"{args.run}__{stamp}.json"
    print(f"Running saved configuration from {source.name} with CURRENT code; this is not an exact historical-code replay. Original file will not be modified.", flush=True)
    record = execute(config)
    record.update(config=config.to_dict(), source_file=source.name,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  note="Saved configuration under current corrected code; not an exact historical-code replay. No attack-strength or optimizer retuning.")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(f"{record['status']}: {output}", flush=True)


if __name__ == "__main__":
    main()
