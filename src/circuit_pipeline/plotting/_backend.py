from __future__ import annotations
import matplotlib

# Safe for CI/headless. Does not override if backend already selected (e.g. notebooks).
matplotlib.use("Agg", force=False)