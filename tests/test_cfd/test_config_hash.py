"""Tests for fanopt.cfd.config_hash (§9.4.1 cross-tier config-hash assertion)."""

from __future__ import annotations

import pytest

from fanopt.cfd import config_hash as ch
from fanopt.cfd.configs import CROSS_TIER


def test_hash_is_stable_24_hex():
    h = ch.cross_tier_config_hash()
    assert h == ch.CROSS_TIER_CONFIG_HASH
    assert len(h) == 24
    assert all(c in "0123456789abcdef" for c in h)


def test_mach_not_in_cross_tier():
    # Round-9 HIGH-12 lock: MACH is tier-specific, must not be in the hashed dict.
    assert "mach_number" not in CROSS_TIER
    assert "mach" not in CROSS_TIER


def test_assert_config_hash_passes_on_locked():
    ch.assert_config_hash(ch.CROSS_TIER_CONFIG_HASH)  # must not raise


def test_assert_config_hash_raises_on_drift():
    with pytest.raises(ValueError, match="numerics drift"):
        ch.assert_config_hash("deadbeefdeadbeefdeadbeef")


def test_design_consistent_passes_when_all_locked():
    ch.assert_design_config_consistent([ch.CROSS_TIER_CONFIG_HASH] * 3)  # must not raise


def test_design_consistent_raises_on_mismatch():
    with pytest.raises(ValueError, match="cross-tier config drift"):
        ch.assert_design_config_consistent([ch.CROSS_TIER_CONFIG_HASH, "0" * 24])


def test_design_consistent_raises_on_empty():
    with pytest.raises(ValueError, match="no config hashes"):
        ch.assert_design_config_consistent([])
