# Seed manifest

Every training run of the paper is identified by a `(seed, weight_init_seed)`
pair, both baked into its `experiments/training/configs/repro_<run_name>.yaml`
and recorded in its frozen `experiments/<figure>/runs/<run>/hydra/config.yaml`.

## Seed semantics (what determines a run exactly)

- **`seed` — process RNG.** With `seed_data_stream: true`, every data batch
  key is `jax.random.fold_in(jax.random.PRNGKey(seed), index)` where `index`
  is the training step (train batches), `num_steps` (held-out eval batch), or
  `num_steps + 1` (end-of-run analysis batch). `seed + 99999` additionally
  seeds the Bayes-optimal-loss Monte-Carlo estimate (N = 32768 samples).
  Data generation is fully deterministic given `(seed, index)`.
- **`weight_init_seed` — model init.** Passed to
  `transformer_lens.HookedTransformerConfig.seed`; determines the initial
  weights. No other stochasticity exists in the training loop (no dropout,
  no data shuffling).
- **`seed_data_stream` (top-level flag, default `false`)** selects between
  two data streams:
  - *step-keyed stream* (flag off): batch keys are `PRNGKey(index)` with no
    seed folding, so the data stream is identical across runs and
    independent of `seed`. Used by the seed-77 canonical runs.
  - *per-seed stream* (flag on): the `fold_in` keys above, so each seed sees
    distinct process samples. Used by every other run.

Every run: 50 000 optimizer steps, batch 25 000 (8192 for the N=5 telephone
family), the log-spaced checkpoint schedule (78 checkpoints), and the config
composition documented in REPRODUCING.md.

## Canonical runs (seed 77 / weight_init_seed 7, step-keyed stream)

`feedback_sns_mess3_canonical`, `feedback_mess3_sns_canonical`,
`feedback_mess3_mess3_canonical`, `telephone_mess3_canonical`,
`telephone_mess3_n2_canonical`, `telephone_mess3_n4_canonical`,
`telephone_mess3_n5_canonical`, `telephone_fractal2_sym_canonical`,
`cg_partition_{full,x,y}_canonical`.

## Additional seeds (`seed = 100·c + r`, `weight_init_seed = seed + 50`, per-seed stream)

| family (c) | runs → (seed, weight_init_seed) |
|---|---|
| feedback_sns_mess3 (1) | s101→(101,151) … s104→(104,154) |
| feedback_mess3_sns (2) | s201→(201,251) … s204→(204,254) |
| telephone_mess3 (4) | s401→(401,451) … s404→(404,454) |
| telephone_mess3_n2 (5) | s501→(501,551) … s504→(504,554) |
| telephone_mess3_n4 (6) | s601→(601,651) … s604→(604,654) |
| telephone_mess3_n5 (7) | s701→(701,751) … s704→(704,754) |
| telephone_fractal2_sym (8) | s801→(801,851) … s804→(804,854) |
| feedback_mess3_mess3 (9) | s901→(901,951) … s904→(904,954) |
| feedback_fractal2_mess3 (10) | s1001→(1001,1051) … s1004→(1004,1054) |
| feedback_mess3_fractal2 (13) | s1301→(1301,1351) … s1304→(1304,1354) |
| telephone_mess3_d60 (20) | s2001→(2001,2051) … s2005→(2005,2055) |
| telephone_mess3_d240 (21) | s2101→(2101,2151) … s2103→(2103,2153) |

Other per-seed-stream runs:

- `feedback_fractal2_mess3_canonical` — seed 1101 / 1151 and
  `feedback_mess3_fractal2_canonical` — seed 1102 / 1152.
- Coarse-graining replicates `cg_r{1..4}_{full,x,y}_canonical` — seeds
  901–912, weight_init_seed = seed + 50 (each replicate's three observation
  modes have their own seeds).

## Width sweep

d_model 60 has five seeds (s2001–s2005), d_model 240 three (s2101–s2103);
the d_model 120 column is the `telephone_mess3_canonical` family (seed 77 +
s401–s404). `experiments/fig13_width_sweep/results.json` caches these 13 runs.
