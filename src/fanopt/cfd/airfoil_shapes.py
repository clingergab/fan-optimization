"""Pure-Python airfoil geometry utilities.

The gmsh-using mesh generator (``scripts/gen_benchmark_meshes.py``) consumes
these helpers as inputs. They live here (importable without gmsh) so:

1. Tests for the shape math run in CI without requiring the gmsh wheel.
2. Future generative-design code can reuse the airfoil discretization
   without pulling in the mesh-generation dependency tree.

Currently supplies NACA 4-digit symmetric airfoils (just 0012 for Spike 0.6c;
the formula generalizes trivially). Cosine spacing along the chord clusters
points near the leading + trailing edges.
"""
from __future__ import annotations

import math

__all__ = [
    "NACA0012_T",
    "NACA0012_CHORD",
    "naca0012_y",
    "airfoil_polyline",
]


NACA0012_T: float = 0.12
"""Thickness ratio (t/c) for NACA 0012."""

NACA0012_CHORD: float = 1.0
"""Default chord (m). ``reynolds_length`` in the benchmark cfg matches this."""


def naca0012_y(
    x: float,
    t: float = NACA0012_T,
    chord: float = NACA0012_CHORD,
) -> float:
    """Half-thickness y(x) on a NACA 4-digit symmetric airfoil's upper surface.

    Lower surface is the negation. Uses the standard NACA 4-digit polynomial
    with the closed-trailing-edge variant (last coefficient -0.1036 instead
    of the open -0.1015) so the polyline returned by ``airfoil_polyline``
    closes exactly at x = chord.

    Parameters
    ----------
    x : chordwise coordinate, in [0, chord].
    t : thickness ratio (t/c). Default 0.12 (NACA 0012).
    chord : chord length, m. Default 1.0.

    Raises
    ------
    ValueError if x is outside [0, chord].
    """
    if x < 0.0 or x > chord:
        raise ValueError(f"x must be in [0, chord={chord}], got {x}")
    xc = x / chord
    return (t / 0.2) * chord * (
        0.2969 * math.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1036 * xc**4
    )


def airfoil_polyline(
    n: int,
    chord: float = NACA0012_CHORD,
) -> list[tuple[float, float]]:
    """Discretize the airfoil into a closed polyline with cosine spacing.

    Returns a list of (x, y) tuples starting at the trailing edge, going
    forward along the upper surface to the leading edge, then back along
    the lower surface to the trailing edge. Cosine spacing denses points
    near LE and TE (where curvature is highest).

    Parameters
    ----------
    n : number of points along each surface (upper, lower). Total polyline
        length is roughly 2n − 2 after de-duplicating LE / TE.
    chord : chord length, m.

    Raises
    ------
    ValueError if n < 16 (too coarse for a usable airfoil).
    """
    if n < 16:
        raise ValueError(f"n must be ≥ 16 for a usable NACA 0012, got {n}")
    beta = [math.pi * i / (n - 1) for i in range(n)]
    xs = [chord * 0.5 * (1.0 - math.cos(b)) for b in beta]
    upper = [(x, naca0012_y(x, chord=chord)) for x in xs]
    lower = [(x, -naca0012_y(x, chord=chord)) for x in reversed(xs)]
    # Drop the duplicated LE / TE between the two halves.
    return list(reversed(upper)) + lower[1:-1]
