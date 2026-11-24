"""Tests for making a blade thinner about its mid-surface (fanopt.geometry.thickness_squeeze).

A closed box slab (x 0..40 mm, y 0..20 mm) whose top and bottom faces can be tilted, so the mid-surface and
the thickness are known by hand at every point.
"""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.geometry.thickness_squeeze import (
    cap_thickness_in_overlap,
    overlap_region,
    squeeze_toward_mid_surface,
    surface_z_grid,
)

MM = 1e-3


def _slab(thickness_m=3 * MM, slope=0.0, step_m=2 * MM, x0=0.0, y0=0.0):
    """Top + bottom sheets (z = slope·x and slope·x + thickness) + side walls, as one closed surface."""
    xs = x0 + np.arange(0.0, 40 * MM + 1e-9, step_m)
    ys = y0 + np.arange(0.0, 20 * MM + 1e-9, step_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nx, ny = gx.shape
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    base = slope * (flat[:, 0] - x0)
    nodes = np.vstack([np.column_stack([flat, base]), np.column_stack([flat, base + thickness_m])])
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            tris += [[a, b, c], [a, c, d]]
    tris = np.array(tris)
    k = len(flat)
    ring = [i * ny for i in range(nx)] + [(nx - 1) * ny + j for j in range(1, ny)]
    ring += [i * ny + ny - 1 for i in range(nx - 2, -1, -1)] + list(range(ny - 2, 0, -1))
    walls = [[p, q, q + k] for p, q in zip(ring, ring[1:] + ring[:1], strict=True)]
    walls += [[p, q + k, p + k] for p, q in zip(ring, ring[1:] + ring[:1], strict=True)]
    return nodes, np.vstack([tris, tris + k, walls])


def _thickness_at(nodes, facets, x, y):
    xs, ys, top, bot = surface_z_grid(nodes, facets, 0.5 * MM)
    i, j = np.argmin(abs(xs - x)), np.argmin(abs(ys - y))
    return top[i, j] - bot[i, j], (top[i, j] + bot[i, j]) / 2


def test_grid_reads_the_top_and_bottom_faces():
    assert _thickness_at(*_slab(), 20 * MM, 10 * MM) == pytest.approx((3 * MM, 1.5 * MM))


def test_grid_is_nan_off_the_part():
    _xs, _ys, top, _bot = surface_z_grid(*_slab(), 0.5 * MM)
    assert np.isnan(top[0, 0])


def test_squeeze_scales_the_thickness():
    nodes, facets = _slab()
    thin = squeeze_toward_mid_surface(nodes, facets, 0.8)
    assert _thickness_at(thin, facets, 20 * MM, 10 * MM)[0] == pytest.approx(2.4 * MM)


def test_squeeze_keeps_the_mid_surface():
    nodes, facets = _slab()
    thin = squeeze_toward_mid_surface(nodes, facets, 0.8)
    assert _thickness_at(thin, facets, 20 * MM, 10 * MM)[1] == pytest.approx(1.5 * MM)


def test_squeeze_keeps_a_tilted_mid_surface():
    nodes, facets = _slab(slope=0.25)
    thin = squeeze_toward_mid_surface(nodes, facets, 0.75)
    t, mid = _thickness_at(thin, facets, 30 * MM, 10 * MM)
    assert (t, mid) == (pytest.approx(2.25 * MM), pytest.approx(0.25 * 30 * MM + 1.5 * MM))


def test_squeeze_keeps_the_outline():
    nodes, facets = _slab()
    thin = squeeze_toward_mid_surface(nodes, facets, 0.8)
    assert np.array_equal(thin[:, :2], nodes[:, :2])


def test_factor_one_is_identity():
    nodes, facets = _slab()
    assert np.array_equal(squeeze_toward_mid_surface(nodes, facets, 1.0), nodes)


def test_input_nodes_are_not_modified():
    nodes, facets = _slab()
    before = nodes.copy()
    squeeze_toward_mid_surface(nodes, facets, 0.8)
    assert np.array_equal(nodes, before)


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.2])
def test_factor_outside_zero_one_is_rejected(factor):
    with pytest.raises(ValueError, match="thickness factor"):
        squeeze_toward_mid_surface(*_slab(), factor)


def test_surface_with_only_side_walls_is_rejected():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1.0]]) * MM
    with pytest.raises(ValueError, match="no faces"):
        squeeze_toward_mid_surface(nodes, np.array([[0, 1, 2]]), 0.8)


# A slab at x 100..140 mm, y -10..10 mm; turned 0.1 rad about the z axis it shifts ~12 mm sideways, so a
# neighbour turned either way covers its two long edges but not its middle.
ANGLE = 0.1


def _offset_slab(thickness_m=3 * MM):
    return _slab(thickness_m=thickness_m, x0=100 * MM, y0=-10 * MM)


def _region_at(x, y):
    xs, ys, top, _bot = surface_z_grid(*_offset_slab(), 0.5 * MM)
    region = overlap_region(xs, ys, ~np.isnan(top), ANGLE)
    return bool(region[np.argmin(abs(xs - x)), np.argmin(abs(ys - y))])


def test_overlap_region_covers_the_edges():
    assert _region_at(120 * MM, 9 * MM) and _region_at(120 * MM, -9 * MM)


def test_overlap_region_leaves_the_middle():
    assert not _region_at(120 * MM, 0.0)


@pytest.fixture(scope="module")
def capped():
    nodes, facets = _offset_slab()
    return cap_thickness_in_overlap(nodes, facets, 2.5 * MM, ANGLE, blend_m=2 * MM), facets


def test_cap_thins_the_overlap_to_the_cap(capped):
    assert _thickness_at(*capped, 120 * MM, 9 * MM)[0] == pytest.approx(2.5 * MM)


def test_cap_keeps_the_mid_surface_in_the_overlap(capped):
    assert _thickness_at(*capped, 120 * MM, 9 * MM)[1] == pytest.approx(1.5 * MM)


def test_cap_leaves_the_middle_alone(capped):
    assert _thickness_at(*capped, 120 * MM, 0.0)[0] == pytest.approx(3 * MM)


def test_cap_leaves_thinner_blades_alone():
    nodes, facets = _offset_slab(thickness_m=2 * MM)
    assert np.allclose(cap_thickness_in_overlap(nodes, facets, 2.5 * MM, ANGLE), nodes)


def test_cap_blends_back_outside_the_overlap():
    nodes, facets = _offset_slab()
    out = cap_thickness_in_overlap(nodes, facets, 2.5 * MM, ANGLE, blend_m=20 * MM)
    t = _thickness_at(out, facets, 120 * MM, 0.0)[0]
    assert 2.5 * MM < t < 3 * MM


def test_cap_must_be_positive():
    with pytest.raises(ValueError, match="cap must be positive"):
        cap_thickness_in_overlap(*_offset_slab(), 0.0, ANGLE)
