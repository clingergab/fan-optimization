"""Direction-lock fold behaviour: base→tip (radial) zigzags nest through the fold swing.

Operator requirement (2026-07-29): zigzag/stairs designs must run BASE→TIP (radial), not
side-to-side (tangential). These tests use the authoritative swept-volume CAD gate
(:func:`fanopt.geometry.blade_cad.fold_collision_clear`) to confirm that both channels the
optimizer uses to build a radial zigzag — the rib-bow meridian and a radial-row panel-offset
grid — actually FOLD CLEAR. Skipped at module load when CadQuery isn't installed, per CLAUDE.md.

Note on the tangential side: under the current z-envelope fold model, layer spacing scales with
each blade's out-of-plane spread, so a side-to-side panel wave within the (small)
``PANEL_OFFSET_MAX_M`` cap still clears the swept-volume check — the direction asymmetry is
enforced by the parameterization (large fold-safe amplitude is only reachable via the radial
meridian; the tangential offset channel is capped small), and locked structurally by the
``displacement_at`` axis-convention tests in ``test_blade.py`` / ``classify_panel`` in
``test_campaign_analysis.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.geometry.blade import BladeParams
from fanopt.geometry.blade_cad import fold_collision_clear


def _flat_panel(t: float = 0.004):
    return tuple((t, t, t) for _ in range(4))


def test_radial_meridian_zigzag_folds_clear():
    # A base→tip meridian zigzag (linear interp, alternating knots) nests through the fold swing —
    # the fold-safe radial shape lever the redesign relies on.
    p = BladeParams(
        blade_count=12,
        rib_bow_knots_m=(0.0, 0.018, 0.0, 0.018, 0.0),
        rib_bow_interp="linear",
        t_rib_hub_m=0.004,
        t_rib_tip_m=0.004,
        panel_offsets_m=tuple((0.0, 0.0, 0.0) for _ in range(4)),
        panel_thickness_m=_flat_panel(0.003),
    )
    assert fold_collision_clear(p) is True


def test_radial_row_panel_zigzag_folds_clear():
    # A panel-offset zigzag that alternates across the 4 RADIAL ROWS (constant across the 3
    # tangential cols) is a base→tip zigzag and must fold clear (uniform sheet, edges unpinned).
    a = 0.0043  # just inside PANEL_OFFSET_MAX_M (~4.5 mm)
    radial_rows = ((a, a, a), (-a, -a, -a), (a, a, a), (-a, -a, -a))
    p = BladeParams(
        blade_count=12,
        rib_bow_knots_m=(0.0, 0.0, 0.0, 0.0, 0.0),
        rib_bow_interp="linear",
        t_rib_hub_m=0.0035,
        t_rib_tip_m=0.0035,
        panel_offsets_m=radial_rows,
        panel_thickness_m=_flat_panel(0.0035),
        uniform=True,
    )
    assert fold_collision_clear(p) is True
