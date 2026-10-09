"""
Central configuration for the DP-FL-IDS data pipeline.

All paths are resolved relative to the project root (the directory that
contains ``src/``), so the pipeline keeps working regardless of where the
project folder is moved. This deliberately replaces the old hard-coded
``/home/ubuntu/Projects/THESIS/musa/...`` paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
# config.py lives at: <project>/code/src/data/config.py
#   parents[2] == <project>/code   (the import root — all Python lives here)
#   parents[3] == <project>        (where data/, data_full/ and results/ live)
CODE_ROOT: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# The raw CICIoT2023 CSVs. The client receives these on a hard drive, so the
# location is overridable via the DPFLIDS_RAW_DIR environment variable (e.g.
# ``E:\CICIoT2023`` on Windows); otherwise fall back to the in-repo location.
_RAW_ENV = os.environ.get("DPFLIDS_RAW_DIR")
RAW_DIR: Path = (Path(_RAW_ENV) if _RAW_ENV
                 else PROJECT_ROOT / "archive" / "wataiData" / "csv" / "CICIoT2023")
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
PARTITION_DIR: Path = PROJECT_ROOT / "data" / "partitions"
ARTIFACT_DIR: Path = PROJECT_ROOT / "data" / "artifacts"  # scaler, encoders, stats

SEED: int = 42

# ── CICIoT2023 schema ────────────────────────────────────────────────────
# 46 numeric network-flow features + 1 label column (verified from raw CSVs).
NETWORK_FEATURES: list[str] = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count", "HTTP",
    "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]
LABEL_COLUMN: str = "label"

# The 34 fine-grained attack labels (exact strings, verified from raw data).
ATTACK_LABELS: list[str] = sorted([
    "BenignTraffic",
    "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood", "DDoS-SYN_Flood",
    "DDoS-PSHACK_Flood", "DDoS-RSTFINFlood", "DDoS-SynonymousIP_Flood",
    "DDoS-HTTP_Flood", "DDoS-SlowLoris", "DDoS-ACK_Fragmentation",
    "DDoS-ICMP_Fragmentation", "DDoS-UDP_Fragmentation",
    "DoS-UDP_Flood", "DoS-TCP_Flood", "DoS-SYN_Flood", "DoS-HTTP_Flood",
    "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain",
    "Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep",
    "Recon-PortScan", "VulnerabilityScan",
    "MITM-ArpSpoofing", "DNS_Spoofing",
    "SqlInjection", "CommandInjection", "XSS",
    "Backdoor_Malware", "BrowserHijacking", "Uploading_Attack",
    "DictionaryBruteForce",
])
assert len(ATTACK_LABELS) == 34, "Expected exactly 34 attack labels"

# 8 coarse categories (7 attack families + Benign), used for the non-IID
# partition design and for category-level evaluation summaries.
CATEGORIES: list[str] = [
    "Benign", "DDoS", "DoS", "Mirai", "Recon", "MITM_Spoof",
    "Injection", "Malware_BF",
]


def label_to_category(label: str) -> str:
    """Map one of the 34 fine labels to its coarse attack category."""
    l = label.lower()
    if "benign" in l:
        return "Benign"
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if l.startswith("mirai"):
        return "Mirai"
    if l.startswith("recon") or "scan" in l or "sweep" in l:
        return "Recon"
    if "mitm" in l or "spoofing" in l:
        return "MITM_Spoof"
    if "injection" in l or "xss" in l:
        return "Injection"
    return "Malware_BF"  # backdoor, hijack, upload, brute force


# ── Synthetic 6G RAN telemetry features (7 → combined 53-dim vector) ──────
# RSRQ is the 7th metric, added so the merged taxonomy is 46 + 7 = 53-dim,
# matching the finalised proposal/presentation.
RAN_FEATURES: list[str] = [
    "RSRP_dBm", "SINR_dB", "RSRQ_dB", "PRB_Util_percent",
    "Handover_Rate_hr", "Beam_Failure_Count", "Throughput_Mbps",
]

# Final combined feature order fed to the model (53 numeric features).
ALL_FEATURES: list[str] = NETWORK_FEATURES + RAN_FEATURES
assert len(ALL_FEATURES) == 53, "Expected 46 network + 7 RAN = 53 features"

# ── Subsampling ──────────────────────────────────────────────────────────
# The full dataset is ~46.7M rows / 13 GB — infeasible to train FL on here.
# We take a stratified per-class subsample: cap the abundant classes, keep
# all rows of the rare classes. Tune MAX_PER_CLASS to trade size vs. realism.
MAX_PER_CLASS: int = 15_000     # cap per fine label
MIN_PER_CLASS: int = 0          # keep everything available for rare classes
CHUNK_SIZE: int = 200_000       # streaming chunk size for pd.read_csv

# ── Non-IID partitioning ─────────────────────────────────────────────────
NUM_CLIENTS: int = 10           # 10 gNBs
DIRICHLET_ALPHA: float = 0.3    # lower = more heterogeneous (more non-IID)
TEST_FRACTION: float = 0.2      # held-out global test split
