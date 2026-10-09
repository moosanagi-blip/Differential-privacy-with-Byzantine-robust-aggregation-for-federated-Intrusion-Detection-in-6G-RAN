"""
Console progress reporting with RO-keyed banners and step markers.

Gives the client the "watch it run" experience requested: each phase prints a
Research-Objective banner and each step prints a start line and a green ✓ with
its wall-clock time. Uses ``colorama`` so ANSI colours work on Windows; degrades
to plain ASCII if colours or a UTF-8 terminal are unavailable.
"""

from __future__ import annotations

import sys
import textwrap
import time
from contextlib import contextmanager

try:
    import colorama
    colorama.init()
    _C = {
        "reset": colorama.Style.RESET_ALL, "bold": colorama.Style.BRIGHT,
        "green": colorama.Fore.GREEN, "cyan": colorama.Fore.CYAN,
        "yellow": colorama.Fore.YELLOW, "red": colorama.Fore.RED,
        "blue": colorama.Fore.BLUE, "magenta": colorama.Fore.MAGENTA,
    }
except Exception:  # pragma: no cover
    _C = {k: "" for k in ("reset", "bold", "green", "cyan", "yellow", "red", "blue", "magenta")}

# Use plain ASCII markers if the terminal can't encode the fancy ones.
_UTF = (getattr(sys.stdout, "encoding", "") or "").lower().startswith("utf")
_OK = "✓" if _UTF else "[OK]"
_ARROW = "▸" if _UTF else ">"
_BAR = "═" if _UTF else "="


# Typographic characters that a non-UTF-8 terminal renders as "?". The
# narration is full of em dashes, so degrade them rather than mangle them.
_TRANSLIT = {
    "—": "-", "–": "-", "…": "...", "‘": "'",
    "’": "'", "“": '"', "”": '"', "×": "x",
    "±": "+/-", "→": "->", "≥": ">=", "≤": "<=",
    "α": "alpha", "ε": "epsilon", "σ": "sigma", "δ": "delta",
}


def _safe(text: str) -> str:
    """Make text printable on the current terminal."""
    if _UTF:
        return text
    for a, b in _TRANSLIT.items():
        text = text.replace(a, b)
    return text.encode("ascii", "replace").decode("ascii")


def _c(text: str, color: str) -> str:
    return f"{_C.get(color, '')}{_safe(text)}{_C['reset']}"


def banner(title: str, color: str = "cyan") -> None:
    line = _BAR * min(72, max(24, len(title) + 4))
    print("\n" + _c(line, color))
    print(_c(f"  {title}", color) if not _C["bold"] else _c(_C["bold"] + f"  {title}", color))
    print(_c(line, color))


def ro(ro_num: int, title: str) -> None:
    """Research-Objective banner, e.g. 'RO1 ▸ Framework …'."""
    banner(f"RO{ro_num} {_ARROW} {title}", color="magenta")


def info(msg: str) -> None:
    print(_c("  • ", "blue") + _safe(msg))


def metric(name: str, value) -> None:
    if isinstance(value, float):
        value = f"{value:.4f}"
    print(_c("    - ", "yellow") + _safe(f"{name}: ") + _c(str(value), "bold"))


def warn(msg: str) -> None:
    print(_c("  ! ", "red") + _safe(msg))


@contextmanager
def step(msg: str):
    """Context manager: prints 'msg …' then '✓ msg (t s)' on success."""
    print(_c(f"  {_ARROW} ", "cyan") + _safe(msg) + _safe(" …"))
    t0 = time.time()
    try:
        yield
    except Exception:
        print(_c(f"  x {_safe(msg)} (failed)", "red"))
        raise
    dt = time.time() - t0
    print(_c(f"  {_OK} ", "green") + _safe(msg) + _c(f"  ({dt:.1f}s)", "yellow"))


# ── presentation helpers (used by the narrated demo) ───────────────────────
WIDTH = 78


def rule(color: str = "cyan") -> None:
    print(_c(_BAR * WIDTH, color))


def say(*paragraphs: str) -> None:
    """Print wrapped explanation paragraphs. Blank line after each."""
    for p in paragraphs:
        print(_safe(textwrap.fill(" ".join(p.split()), width=WIDTH,
                                  initial_indent="  ", subsequent_indent="  ")))
        print()


def takeaway(msg: str) -> None:
    """Highlight the conclusion of a step."""
    body = textwrap.fill(">> TAKEAWAY: " + " ".join(msg.split()), width=WIDTH,
                         initial_indent="  ", subsequent_indent="     ")
    print(_c(body, "green"))
    print()


def substep(msg: str) -> None:
    """A live progress line nested under a step (e.g. one gNB training)."""
    print(_c(f"      {_ARROW} ", "blue") + _safe(msg))


def stdin_is_interactive() -> bool:
    """Whether there is a human at a keyboard to wait for."""
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def prompt(msg: str) -> None:
    """Wait for Enter. Returns immediately if stdin is not a terminal, so
    redirected and piped runs do not block forever. Ctrl-C still aborts."""
    if not stdin_is_interactive():
        return
    try:
        input(_c(f"  {msg}", "cyan"))
    except EOFError:
        print()


def kv(pairs: dict, indent: str = "    ") -> None:
    """Aligned key/value block."""
    if not pairs:
        return
    w = max(len(str(k)) for k in pairs)
    for k, v in pairs.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        print(_c(f"{indent}{str(k):<{w}}", "yellow") + "  " + _c(str(v), "bold"))


def table(headers: list[str], rows: list[list], indent: str = "  ",
          aligns: str | None = None) -> None:
    """Fixed-width text table.

    ``aligns`` is one char per column: '<' left, '>' right. Defaults to left for
    the first column and right for the rest, which suits metric tables.
    """
    cells = [[("" if c is None else str(c)) for c in r] for r in rows]
    ncol = len(headers)
    aligns = aligns or ("<" + ">" * (ncol - 1))
    widths = [len(str(headers[i])) for i in range(ncol)]
    for r in cells:
        for i in range(ncol):
            widths[i] = max(widths[i], len(r[i]) if i < len(r) else 0)

    head = indent + "  ".join(
        f"{str(headers[i]):{aligns[i]}{widths[i]}}" for i in range(ncol))
    print(_c(head, "bold"))
    print(indent + "  ".join("-" * widths[i] for i in range(ncol)))
    for r in cells:
        print(_safe(indent + "  ".join(
            f"{(r[i] if i < len(r) else ''):{aligns[i]}{widths[i]}}"
            for i in range(ncol))))
    print()
