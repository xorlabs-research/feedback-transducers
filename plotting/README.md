# plotting/ — reusable belief-geometry analysis + figure library

Parameterized by run name; run dirs are resolved as
`$FACTORED_REPS_DATA/experiments/<figure>/runs/<run>/` (default: this repo), each
containing the frozen `hydra/config.yaml` plus `checkpoints/<step>/model.pt`
and (for loss curves) `mlruns/`.

Paper-figure modules (see the notebooks under `notebooks/` for the drivers):

- `lib/feedback_runs.py` — shared loaders + probe/PCA/CEV helpers for the
  feedback-transducer runs (fixed probe batch: 1024 sequences, PRNGKey(12345)).
- `feedback_msp.py` — feedback MSP reconstruction, training timelines, and the
  belief-patching KL column/bars (pair="fractal2" is the paper's).
- `steering_kl_controls.py` — the four-KL patching bars
  (reads `analysis/steering_kl_controls_paper.json`).
- `dimension_cells.py` — the dimensional-advantage figure (CEV cell, rank-vs-N,
  coarse-graining panel; LaTeX-assembled with the (a,b) diagram panels) and the
  appendix fractal2 CEV cell.
- `telephone_cev.py` — telephone CEV panels with across-seed min–max bands.
- `effective_rank_vs_n.py` — effective rank vs number of sub-agents N.
- `telephone_combined.py` — rank_points()/RANK_RUNS: the rank-vs-N data for the dimension figure.
- `probe_controls.py` — probe selectivity + layerwise emergence
  (reads the committed probe JSONs under `experiments/fig11_probe_controls/`).
- `orth_comparison.py` — steering-write orthogonalization comparison
  (reads `analysis/steering_orth_variants.json`).
- `convergence_curves.py` — eval loss vs Bayes floor from the mlflow stores
  (the paper uses the symlog-x variant).
- `coarse_graining_figure.py` — coarse-graining CEV comparison from
  `analysis/coarse_graining_trained.json`.
- `lib/eval_lib.py`, `lib/runs.py` — mlflow parsing, run loading, probes.

The width-sweep figure (`effective_rank_vs_width`) and the dose-response
figure live with their data under `experiments/fig13_width_sweep/` and
`experiments/fig17_dose_response/`.
