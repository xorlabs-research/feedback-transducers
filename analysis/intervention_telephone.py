"""The two telephone seed families used by the belief-patching pipelines."""
from __future__ import annotations

FAMILIES = {
    "telephone": ["telephone_mess3_canonical_50k"]
    + [f"telephone_mess3_canonical_s{s}_50k" for s in (401, 402, 403, 404)],
    "fractal2_telephone": ["telephone_fractal2_sym_canonical_50k"]
    + [f"telephone_fractal2_sym_canonical_s{s}_50k" for s in (801, 802, 803, 804)],
}
