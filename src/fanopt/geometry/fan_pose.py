"""Pose one blade into the full 12-blade fan — folded (stowed) and deployed (unfolded).

The blades are threaded on the pivot pin (the ``+z`` axis through the origin) at fixed ``layer_spacing``
z-layers; folding does not slide them along the pin, it only rotates them about it. So:

- **FOLDED / stowed**: every blade at rotation 0 — a pure z-stack (the folded deck).
- **DEPLOYED / unfolded**: blade *i* rotated to ``INTER_BLADE_ANGLE_DEG · (i − (N−1)/2)`` about the pin,
  still on its own z-layer — so the open fan is a shallow cone (each blade one layer up the pin).

Operates on plain vertex arrays, so it poses either the carved TO surface (from
:mod:`fanopt.geometry.to_stl`) or any CadQuery-tessellated blade. Pure numpy — no CadQuery, no meshing.
"""

from __future__ import annotations

import numpy as np

from fanopt.geometry.blade import BLADE_COUNT, BladeParams, layer_spacing_m
from fanopt.geometry.schema import INTER_BLADE_ANGLE_DEG

__all__ = ["fan_blade_poses", "apply_pose", "pose_fan"]


def fan_blade_poses(
    blade_count: int, spacing_m: float, *, deployed: bool, direction: int = 1
) -> list[tuple[float, float]]:
    """``(rotation_deg about +z, z_offset_m)`` for each blade in the folded or deployed fan.

    Deployed rotations are centred on 0 (symmetric fan); folded rotations are all 0 (aligned stack).
    Both states place blade *i* at ``z = i · spacing_m`` — the fixed pin layers. ``direction`` (+1/−1)
    flips which way the leaves fan out (which layer sweeps which way) — a fold-symmetric choice that
    only changes the shingle/overlap sense (and comfort), never the per-blade aero.
    """
    return [
        (direction * INTER_BLADE_ANGLE_DEG * (i - (blade_count - 1) / 2.0) if deployed else 0.0, i * spacing_m)
        for i in range(blade_count)
    ]


def apply_pose(verts: np.ndarray, rot_deg: float, z_offset_m: float) -> np.ndarray:
    """Rotate ``verts`` (V,3) about the pin (+z through origin) by ``rot_deg`` and shift z by the offset."""
    theta = np.radians(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.asarray(verts, dtype=float) @ rot.T
    out[:, 2] += z_offset_m
    return out


def pose_fan(
    verts: np.ndarray,
    faces: np.ndarray,
    params: BladeParams,
    *,
    deployed: bool,
    blade_count: int | None = None,
    direction: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pose one blade's ``(verts, faces)`` into the whole fan → N ``(posed_verts, faces)``.

    ``faces`` is shared unchanged across blades (only the vertices move). Both the count and the z-layer
    spacing come from the design: ``blade_count`` defaults to ``params.blade_count`` (so a legacy 8/10-blade
    design poses as itself, not a hard-coded 12), and the spacing is the design's own
    :func:`fanopt.geometry.blade.layer_spacing_m` (blade z-envelope + fold clearance). ``direction`` (+1/−1)
    reverses the deploy fan-out sense (top-vs-bottom layers swap sweep sides) — fold-symmetric, aero-neutral.
    """
    n = getattr(params, "blade_count", BLADE_COUNT) if blade_count is None else blade_count
    spacing = layer_spacing_m(params)
    poses = fan_blade_poses(n, spacing, deployed=deployed, direction=direction)
    return [(apply_pose(verts, rot, z), faces) for rot, z in poses]
