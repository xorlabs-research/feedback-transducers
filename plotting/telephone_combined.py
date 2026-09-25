"""Effective-rank-vs-N points for the telephone N-sweep (panel (d) of the
paper's dimensional-advantage figure; drawn by plotting.dimension_cells).

Only the data collection lives here; the panel itself is drawn by
plotting.effective_rank_vs_n.draw_rank_panel.
"""
from __future__ import annotations

from .effective_rank_vs_n import aggregate_seeds, effective_rank, theory_thresholds

# N -> canonical run + its four seed reruns
RANK_RUNS = {
    2: ["telephone_mess3_n2_canonical_50k"]
       + [f"telephone_mess3_n2_canonical_s{s}_50k" for s in (501, 502, 503, 504)],
    3: ["telephone_mess3_canonical_50k"]
       + [f"telephone_mess3_canonical_s{s}_50k" for s in (401, 402, 403, 404)],
    4: ["telephone_mess3_n4_canonical_50k"]
       + [f"telephone_mess3_n4_canonical_s{s}_50k" for s in (601, 602, 603, 604)],
    5: ["telephone_mess3_n5_canonical_50k"]
       + [f"telephone_mess3_n5_canonical_s{s}_50k" for s in (701, 702, 703, 704)],
}
THRESHOLDS = [0.80, 0.95]
# The rank panel samples 4096 sequences (the CEV panels of the same figure use
# plotting.lib.feedback_runs.BATCH = 1024): the effective rank is a statistic
# of the whole activation cloud and was measured on the larger batch in the
# paper. Same fixed key as everywhere else.
BATCH, SEED = 4096, 12345


def rank_points():
    points = []
    for N, runs in sorted(RANK_RUNS.items()):
        results = []
        for run in runs:
            try:
                results.append(effective_rank(run, BATCH, SEED, THRESHOLDS))
            except FileNotFoundError as e:
                print(f"  SKIP N={N} {run}: {e}")
        if results:
            agg = aggregate_seeds(results, THRESHOLDS)
            # Bayes-belief n_c at the same thresholds, on the same batch;
            # identical across seeds by construction, so the first run suffices
            agg.update(theory_thresholds(runs[0], BATCH, SEED, THRESHOLDS))
            points.append((N, agg))
            print(f"  N={N}: {len(results)} seed(s), nc={agg['nc']}  "
                  f"nc_factored={agg['nc_factored']}  nc_kron={agg['nc_kron']}",
                  flush=True)
    return points
