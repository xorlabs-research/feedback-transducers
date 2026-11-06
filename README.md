# Beliefs in feedback: Factored representations of interacting agents in transformers

Reproduction code for the paper *Beliefs in feedback: Factored representations of interacting agents in transformers*
(David A. Ehrlich, Manuel Baltieri, Fernando E. Rosas, Alexander B. Boyd; [arXiv:2611.02047](https://arxiv.org/abs/2611.02047)).

The code retrains the paper's training runs exactly and regenerates its data figures.

This code builds on and extends https://github.com/Astera-org/factored-reps.
This fork has been created and is actively maintained by David A. Ehrlich (david@daehrlich.com)

## Setup

```bash
uv sync            # or: pip install -e .
```

## Quick tour

```bash
# retrain any run exactly (seeds + params baked into the config)
cd experiments/training
python run.py --config-name repro_feedback_mess3_fractal2_canonical device=cuda

# regenerate the paper's plots (needs checkpoints/mlruns; see REPRODUCING.md)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

- `notebooks/` — one executed notebook per paper figure family, with the
  plots inline.
- `experiments/<figure>/runs/<run>/hydra/` — frozen resolved config of every run as
  trained; `experiments/SEED_MANIFEST.md` — seed semantics.
- `plotting/` — the reusable figure library; `analysis/` — steering and
  coarse-graining pipelines with committed results.

## Cite as

```bibtex
@article{ehrlich2026beliefs,
  title   = {Beliefs in feedback: Factored representations of interacting agents in transformers},
  author  = {Ehrlich, David A. and Baltieri, Manuel and Rosas, Fernando E. and Boyd, Alexander B.},
  journal = {arXiv preprint arXiv:2611.02047},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.02047},
}
```
