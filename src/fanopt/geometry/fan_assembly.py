"""Deployed-fan-level helpers — multi-blade composition + mass / inertia.

Sits on top of :func:`fanopt.geometry.assembly_cad.make_vunit_blade`.
Produces the deployed fan (N blades rotated about the pivot pin axis)
plus the per-design physical properties (mass, centre of mass,
moment of inertia about the wrist axis) that the Phase 4 BO ledger +
the deferred Spike 0.2 V2 cross-check both consume.

Inertia convention
------------------

The wrist axis is the +y axis through the handle grip, located
``D_HANDLE_M`` inboard of the pivot pin at ``x = -D_HANDLE_M`` (per
plan §0 row 27). The rotational inertia about this axis is

    I_wrist = ∫ ρ · ((x − x_wrist)² + z²) dV

with ``x_wrist = -D_HANDLE_M``. Computed via CadQuery's
``Solid.matrixOfInertia()`` (returns the tensor at the centroid with
unit density) + parallel-axis theorem + density multiply.
"""

from __future__ import annotations

import math

import cadquery as cq

from fanopt.geometry.assembly_cad import make_vunit_blade
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.schema import (
    D_HANDLE_M,
    INTER_BLADE_ANGLE_RAD,
    RHO_PETG_KG_PER_M3,
)

__all__ = [
    "deploy_fan",
    "compute_mass_kg",
    "compute_centre_of_mass",
    "compute_i_wrist_kgm2",
]


def deploy_fan(
    params: BladeDesignParams,
) -> tuple[cq.Workplane, tuple[cq.Workplane, ...]]:
    """Return ``(assembly, per_blade_workplanes)`` for the deployed fan.

    Builds one V-unit blade and rotates it about the +z axis at the
    pivot to produce ``params.layer1.blade_count`` instances spanning
    the deployed fan arc. The assembly is a single ``cq.Workplane``
    containing all blades as a Compound; ``per_blade_workplanes`` is
    a tuple of the individual blade Workplanes (useful for per-blade
    STL export).

    The pivot axis is at ``(PIVOT_CENTER_X_M, 0, 0)`` from the V-unit's
    own frame; the blades fan out from there.
    """
    blade = make_vunit_blade(params)
    n = params.layer1.blade_count
    blades: list[cq.Workplane] = []
    for i in range(n):
        # Centre the fan at i = (n-1)/2 (symmetric about y).
        delta_deg = math.degrees(INTER_BLADE_ANGLE_RAD) * (i - (n - 1) / 2.0)
        rotated = blade.rotate((0, 0, 0), (0, 0, 1), delta_deg)
        blades.append(rotated)

    compound = cq.Compound.makeCompound([b.val() for b in blades])
    assembly = cq.Workplane("XY").newObject([compound])
    return assembly, tuple(blades)


def compute_mass_kg(
    blade: cq.Workplane,
    density_kg_per_m3: float = RHO_PETG_KG_PER_M3,
) -> float:
    """Mass of one blade at the given material density."""
    vol_m3 = blade.val().Volume()
    return vol_m3 * density_kg_per_m3


def compute_centre_of_mass(blade: cq.Workplane) -> tuple[float, float, float]:
    """Centroid of one blade as ``(x, y, z)`` in metres."""
    c = blade.val().Center()
    return (c.x, c.y, c.z)


def compute_i_wrist_kgm2(
    blade: cq.Workplane,
    density_kg_per_m3: float = RHO_PETG_KG_PER_M3,
) -> float:
    """Moment of inertia of one blade about the wrist (+y) axis.

    Parallel-axis decomposition:

        I_wrist = ρ · ((I_yy_centroid / ρ_unit) + V · d²)

    where ``I_yy_centroid`` is the y-component of the unit-density
    inertia tensor at the centroid, ``V`` is the volume, and ``d²``
    is the squared in-plane distance from the centroid to the wrist
    axis ``(x = -D_HANDLE_M, z = 0)``.
    """
    solid = blade.val()
    inertia = cq.Shape.matrixOfInertia(solid)
    # CadQuery returns a cq.Matrix; index access goes through its rows.
    # Matrix.__getitem__ returns row-i as a tuple of 4 floats (3x4 affine
    # storage); the [1][1] selection picks the yy component.
    iyy_unit = inertia[1][1]
    c = solid.Center()
    dx = c.x - (-D_HANDLE_M)
    dz = c.z - 0.0
    d2 = dx * dx + dz * dz
    vol = solid.Volume()
    return density_kg_per_m3 * (iyy_unit + vol * d2)
