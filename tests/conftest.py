import matplotlib

# Force a non-interactive backend for CI/tests (no Tk required)
matplotlib.use("Agg", force=True)
