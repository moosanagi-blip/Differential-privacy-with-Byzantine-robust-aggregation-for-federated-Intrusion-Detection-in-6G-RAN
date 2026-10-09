"""Narrated end-to-end demonstration of the DP-FL-IDS framework.

``main.py`` at the project root is the single entry point; everything it prints
comes from here. The package is split so that the *words* can be edited without
touching the *logic*:

- ``narration``     — every explanatory paragraph, one entry per step
- ``steps``         — what each step actually executes
- ``progress``      — turns engine ``on_event`` callbacks into live console output
- ``thesis_numbers``— the published 30-round x 5-seed figures, for side-by-side
- ``runner``        — sequencing, timing, caching, transcript, error containment
- ``replay``        — narrate saved results without training anything
"""
