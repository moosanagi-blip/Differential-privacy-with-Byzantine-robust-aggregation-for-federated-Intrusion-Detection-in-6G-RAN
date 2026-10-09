"""Export paper-ready figures from saved revision-pilot JSON files.

These figures are intentionally labelled as pilots because the available raw
corpus contains 23.37M prepared flows, not the manuscript's claimed 46.69M.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PILOTS = ROOT / "results" / "revision_pilot"
OUT = ROOT.parent / "paper" / "figures" / "revision_pilots"
OUT.mkdir(parents=True, exist_ok=True)


def load(stem: str) -> dict:
    with (PILOTS / f"{stem}.json").open(encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def adaptive_admission() -> None:
    runs = [
        ("Static coalition", load("revision_adaptive_static_5r")),
        ("Rotating coalition", load("revision_adaptive_rotating_5r")),
    ]
    labels = [name for name, _ in runs]
    admitted = [sum(len(t["byz_selected"]) for t in run["timeline"]) for _, run in runs]
    possible = [sum(len(t["byzantine_clients"]) for t in run["timeline"]) for _, run in runs]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    bars = ax.bar(labels, admitted, color=["#b2182b", "#ef8a62"], width=0.58)
    ax.set_ylim(0, max(possible) + 3)
    ax.set_ylabel("Malicious updates admitted")
    ax.set_title("Adaptive poisoning against Multi-Krum (pilot)")
    ax.grid(axis="y", alpha=0.25)
    for bar, num, den in zip(bars, admitted, possible):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 0.35, f"{num}/{den}", ha="center", weight="bold")
    save(fig, "fig_revision_adaptive_admission")


def f_sensitivity() -> None:
    rows = [(f, load(f"revision_festimate_{f}_3r")) for f in range(5)]
    fs = np.array([f for f, _ in rows])
    accuracy = np.array([r["final_accuracy"] for _, r in rows])
    admitted = np.array([sum(len(t["byz_selected"]) for t in r["timeline"]) for _, r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.3, 3.8))
    ax1.plot(fs, accuracy, marker="o", linewidth=2, color="#2166ac")
    ax1.set_xlabel("Configured attacker count $f$")
    ax1.set_ylabel("Final accuracy")
    ax1.set_xticks(fs)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)
    ax1.set_title("Utility")
    bars = ax2.bar(fs, admitted, color="#b2182b", width=0.62)
    ax2.set_xlabel("Configured attacker count $f$")
    ax2.set_ylabel("Malicious updates admitted")
    ax2.set_xticks(fs)
    ax2.set_ylim(0, 10)
    ax2.grid(axis="y", alpha=0.25)
    for bar, num in zip(bars, admitted):
        ax2.text(bar.get_x() + bar.get_width() / 2, num + 0.2, str(num), ha="center")
    fig.suptitle("Sensitivity to estimated attacker count (pilot)", y=1.03)
    save(fig, "fig_revision_f_sensitivity")


def radio_ablation() -> None:
    full = load("revision_radio_full_3r")
    masked = load("revision_radio_masked_3r")
    labels = ["All features", "Synthetic RAN\nfeatures masked"]
    metrics = [
        ("Accuracy", [full["final_accuracy"], masked["final_accuracy"]]),
        ("Macro-F1", [full["final_macro_f1"], masked["final_macro_f1"]]),
        ("False-positive rate", [full["final_fpr"], masked["final_fpr"]]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))
    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=["#2166ac", "#b2182b"], width=0.62)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Dependence on synthetic radio features (pilot)", y=1.04)
    save(fig, "fig_revision_radio_ablation")


if __name__ == "__main__":
    adaptive_admission()
    f_sensitivity()
    radio_ablation()
    print(f"Wrote revision pilot figures to {OUT}")
