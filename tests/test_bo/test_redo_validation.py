"""Tests for the Stage-2 fidelity-agreement and headroom analyses."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo.redo_validation import headroom, ranking_agreement


def test_identical_rankings_preserved():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = ranking_agreement(v, v)
    assert a.kendall_tau == pytest.approx(1.0)
    assert a.ranking_preserved is True


def test_reversed_rankings_not_preserved():
    a = ranking_agreement([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert a.kendall_tau == pytest.approx(-1.0)
    assert a.ranking_preserved is False


def test_agreement_drops_nonfinite_pairs():
    a = ranking_agreement([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert a.n == 3  # the NaN pair dropped


def test_agreement_needs_two_pairs():
    a = ranking_agreement([1.0, np.nan], [np.nan, 2.0])
    assert a.n == 0 and a.ranking_preserved is None


def test_constant_field_has_no_headroom():
    h = headroom([1.0e12] * 6)
    assert h.cv == pytest.approx(0.0)
    assert h.has_headroom is False


def test_varied_field_has_headroom():
    h = headroom([0.5e12, 1.0e12, 1.5e12, 2.0e12])
    assert h.cv > 0.1
    assert h.has_headroom is True


def test_headroom_ignores_nonfinite():
    h = headroom([1.0e12, np.nan, 1.0e12])
    assert h.n == 2 and h.cv == pytest.approx(0.0)
