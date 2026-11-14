"""Root-screen: Adult SA=Y и CMC (две disjoint-проверки) — PASS/FAIL корня.

Только |D_train| >= k и H(B_SA|D_train) >= log2(l); без ML/NCP/атак.
Y используется только для стратификации сплита (воспроизводимость).
"""

from __future__ import annotations

import numpy as np

import core


def root_screen(b_sa_values: np.ndarray, k: int, ell: int) -> dict:
    """PASS/FAIL корня по B_SA на D_train."""
    n = int(len(b_sa_values))
    _, counts = np.unique(b_sa_values, return_counts=True)
    h = core.entropy_bits(counts)
    ok_size = n >= k
    ok_h = h >= np.log2(ell) - 1e-12
    return {
        "n_train": n,
        "h_b_sa": float(h),
        "h_threshold": float(np.log2(ell)),
        "size_ok": bool(ok_size),
        "entropy_ok": bool(ok_h),
        "pass": bool(ok_size and ok_h),
        "reason": "" if (ok_size and ok_h) else
        ("|D_train| < k" if not ok_size else f"H(B_SA)={h:.4f} < log2(l)={np.log2(ell):.4f}"),
    }
