# Reproducing the paper's runs and figures

This repository contains the code needed to (1) retrain the training runs of
*Beliefs in feedback: Factored representations of interacting agents in transformers* exactly and (2) regenerate the paper's eleven data figures,
plus a notebook that does the latter end-to-end.

## Layout

```
fwh_core/                 core library (processes, models, metrics)
experiments/training/     training harness (hydra); one repro_* config per run
experiments/SEED_MANIFEST.md  seed semantics & provenance of every run
experiments/<figure>/     one dir per paper figure (or qualitatively distinct
                          subfigure), numbered as in the manuscript:
                          runs/<run>/ holds the frozen resolved config of
                          each run as trained, next to the figure's cross-run
                          analysis pipelines + committed results
  fig02_feedback_msp/       fractal2-pair loops A/B (also Figs. 3, 9-11, 15-17)
  fig04cd_telephone_mess3/  telephone mess3, N = 2..5 (also Figs. 13, 15, 16)
  fig04e_coarse_graining/   coarse-graining demonstrator runs
  fig11_probe_controls/     probe selectivity + layerwise-emergence pipelines
  fig12_telephone_fractal2/ telephone 5x fractal2 (also Figs. 15, 16)
  fig13_width_sweep/        d_model 60/240 runs + width-sweep analysis
  fig16_orth_comparison/    mess3<->mess3 and mess3-sns loops
  fig17_dose_response/      steering dose-response pipeline
plotting/                 reusable figure library (see plotting/README.md)
analysis/                 intervention/steering pipelines, committed result
                          JSONs, and the diagram assets of the dimension figure
notebooks/                one executed notebook per paper figure family
figures/                  notebook outputs (not tracked)
```

## The eleven paper figures

| manuscript file | built by | data |
|---|---|---|
| `figure_feedback_msp_fractal2.pdf` | `plotting.feedback_msp` (pair=fractal2) | ckpts of the 2 canonical fractal2-pair runs + `analysis/steering_kl_controls_paper.json` |
| `figure_feedback_msp_fractal2_setupA_timeline.pdf` | same | same ckpts |
| `figure_feedback_msp_fractal2_setupB_timeline.pdf` | same | same ckpts |
| `figure_steering_setupA.pdf` | same | `analysis/steering_kl_controls_paper.json` (35 runs, final ckpts) |
| `figure_dimension_full.pdf` | `plotting.dimension_cells` (+ `pdflatex`) | ckpts of 25 telephone runs (5 families × 5 seeds, N=3 shared) + `analysis/coarse_graining_trained.json` + `analysis/{figure_dimension_full_manual.tex,social_network_ab_manual.pdf}` (the hand-drawn (a,b) panels; `figure_dimension_full.tex` + `social_network.tikz` draw them in TikZ if the export is absent) |
| `figure_cell_tel_cev_f2.pdf` | `plotting.dimension_cells` | ckpts of the 5 fractal2-telephone runs |
| `figure_probe_controls.pdf` | `plotting.probe_controls` | committed probe JSONs (10 fractal2-pair runs, 78 + 14 ckpts each) |
| `effective_rank_vs_width.pdf` | `experiments/fig13_width_sweep/analysis.py` | committed `results.json` cache (13 runs) or ckpts |
| `figure_convergence.pdf` | `plotting.convergence_curves` (**symlog-x variant**) | mlflow eval-loss series, 35 runs |
| `figure_orth_comparison.pdf` | `plotting.orth_comparison` | `analysis/steering_orth_variants.json` (27 runs, final ckpts) |
| `figure_dose_response.pdf` | `experiments/fig17_dose_response/figure_dose_response.py --out-dir figures` | committed `dose_response.{json,npz}` (10 runs) |

## Retraining a run (exact)

Every run has a self-contained config whose composition equals the frozen
`experiments/<figure>/runs/<run>/hydra/config.yaml` on every training-critical key
(seed, weight_init_seed, data-stream mode, process spec, model, optimizer,
schedule, steps, batch size):

```bash
cd experiments/training
python run.py --config-name repro_<run_name> device=cuda
# e.g. repro_feedback_mess3_fractal2_canonical_s1301,
#      repro_telephone_mess3_d240_canonical_s2101, repro_cg_r2_x_canonical, ...
```

Determinism: data batches are `PRNGKey(step)` (step-keyed stream, the seed-77
canonical runs) or `fold_in(PRNGKey(seed), step)` (per-seed stream,
`seed_data_stream: true` — all other runs);
weights init from `HookedTransformerConfig.seed = weight_init_seed`; no
dropout or shuffling. See `experiments/SEED_MANIFEST.md` for the full
table.

## Regenerating the figures

Checkpoints and mlflow stores are not tracked. Either drop them into
`experiments/<figure>/runs/<run>/{checkpoints,mlruns}`, or set
`FACTORED_REPS_DATA` to a directory with that same layout, then run the notebook (`figure_dimension_full.pdf` needs
`pdflatex` on the PATH):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

The notebooks draw every DATA subfigure of the paper; the conceptual
diagram panels (a,b) (and the LaTeX composition of `figure_dimension_full.pdf`,
available via `plotting.dimension_cells.make_figures(..., assemble=True)`)
are not part of the notebook outputs.

Figures built from committed intermediate artifacts need no checkpoints at
all (steering bars, orthogonalization comparison, dose response, probe
controls, coarse-graining panel, width sweep via its cache). To rebuild those
artifacts from checkpoints:

```bash
python analysis/steering_kl_controls_paper.py     # steering KL bars (35 runs)
python analysis/steering_orth_variants.py         # orthogonalization variants
python analysis/coarse_graining_trained.py        # coarse-graining CEV
python experiments/fig11_probe_controls/probe_selectivity_controls/probe_selectivity_controls.py
python experiments/fig11_probe_controls/probe_layerwise_emergence/compute_layerwise_r2.py
python experiments/fig13_width_sweep/analysis.py --refresh
python experiments/fig17_dose_response/dose_response.py
```

## Environment

Python 3.12, `uv sync` or `pip install -e .` (see `pyproject.toml`), which
installs `fwh_core`, `experiments` and `plotting` as packages and the
notebook tooling (`nbconvert`); `analysis/` and the `experiments/<figure>/`
pipelines are plain scripts run from the repo root (they add the repo root to
`sys.path` themselves, as do the notebooks). `pdflatex` (TeX Live) is needed
only for the assembled dimension figure. The figure
verification was run with the project's pinned versions of
numpy/torch/jax/transformer_lens/matplotlib; exact figure reproduction
assumes the pinned environment (linear algebra and font rasterization are
library-version dependent).
