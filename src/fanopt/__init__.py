"""fanopt — 3D-printed folding fan TO/ASO toolkit.

Canonical specification: docs/plan_R11.md
Locks-to-location index:  docs/locks_index.md
Adversarial-review log:   docs/reviews/

Subpackages mirror docs/plan_R11.md §12.1:
    geometry  CadQuery generator + §N7 manufacturability filter
    topopt    Rib SIMP / Reissner-Mindlin plate-bending TO
    cfd       SU2 wrappers + canonical J_fan post-processor
    bo        Multi-fidelity Bayesian optimization (BoTorch)
    physical  Post-processing of physical measurements (IMU, acoustic, anemometer)
    utils     Ledger / Drive IO / Colab session helpers / structured logging
"""
from __future__ import annotations

__version__ = "0.1.0a0"
__all__ = ["__version__"]
