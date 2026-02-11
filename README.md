# Circuit Pipeline — neural circuit analysis (grid + plots + manifest)

This repository is a small, reproducible pipeline for analyzing a 4-population neural circuit model.
It runs a parameter grid, saves results to `.npz`, generates plots, and records metadata in a `manifest.json`.

## What you can do in one command

After installation you can run:

```bash
circuit_grid --config configs/reference_dynamics.yaml --run-root runs --progress
