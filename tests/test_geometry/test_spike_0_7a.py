"""Library-level tests for ``fanopt.geometry.spike_0_7a``.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7`` (sub-spike 0.7a).
Library docstring: ``src/fanopt/geometry/spike_0_7a.py``.

These tests exercise the library half (parameter sampling, record dataclass,
analyze gate) without touching the CadQuery pipeline. The four pipeline
callables are mocked with deterministic fakes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fanopt.geometry.spike_0_7a import (
    ADVERSARIAL_PARAM_SETS,
    RANDOM_DOCUMENTED_FAIL_GATE,
    SCHEMA_BOUNDS,
    GeomSanityRecord,
    Spike07aResult,
    analyze_07a,
    evaluate_param_set,
    hash_params,
    random_param_set_within_bounds,
)


# ── Fake pipeline callables ─────────────────────────────────────────────


def _fake_gen_ok(params: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "fake_blade", "params_hash": hash_params(params)}


def _fake_gen_none(_params: dict[str, Any]) -> None:
    return None


def _fake_pass(_params: dict[str, Any], _blade: Any) -> tuple[bool, tuple[str, ...]]:
    return True, ()


def _fake_fail(reason: str):
    def _f(_p: dict[str, Any], _b: Any) -> tuple[bool, tuple[str, ...]]:
        return False, (reason,)

    return _f


# ── Random-set sampling ─────────────────────────────────────────────────


def test_random_param_sets_respect_schema_bounds() -> None:
    """All numeric fields in every random draw must lie within their
    declared bounds; every categorical draw must come from its choice list;
    every boolean is in {True, False}."""
    rng = np.random.default_rng(123)
    sets = random_param_set_within_bounds(rng, 25)
    assert len(sets) == 25

    for s in sets:
        assert s["is_adversarial"] is False
        for name, spec in SCHEMA_BOUNDS.items():
            assert name in s, f"random draw missing key {name}"
            v = s[name]
            kind = spec[0]
            if kind == "float":
                lo, hi = spec[1], spec[2]
                assert isinstance(v, float), f"{name} should be float, got {type(v)}"
                # Allow lo == hi (locked params).
                assert lo - 1e-12 <= v <= hi + 1e-12, f"{name}={v} outside [{lo},{hi}]"
            elif kind == "int":
                lo, hi = spec[1], spec[2]
                assert isinstance(v, int)
                assert lo <= v <= hi
            elif kind == "choice":
                choices = list(spec[1])
                assert v in choices, f"{name}={v!r} not in {choices}"
            elif kind == "bool":
                assert isinstance(v, bool)


def test_random_param_set_reproducible_with_same_seed() -> None:
    a = random_param_set_within_bounds(np.random.default_rng(7), 5)
    b = random_param_set_within_bounds(np.random.default_rng(7), 5)
    assert a == b


def test_random_param_set_rejects_zero_n() -> None:
    with pytest.raises(ValueError, match="must be"):
        random_param_set_within_bounds(np.random.default_rng(0), 0)


# ── Adversarial-set sanity ──────────────────────────────────────────────


def test_adversarial_param_sets_count_at_least_3() -> None:
    """Spec sub-clause requires ≥ 3 hand-picked adversarial sets."""
    assert len(ADVERSARIAL_PARAM_SETS) >= 3
    for p in ADVERSARIAL_PARAM_SETS:
        assert p.get("is_adversarial") is True
        assert "_adversarial_id" in p
        assert "_adversarial_target" in p


def test_adversarial_param_sets_include_louver_tpms_primitive() -> None:
    """Spec sub-clause names three specific adversarial categories:
        (a) Layer 2 louver clustered at tip
        (b) Layer 2 TPMS at minimum cell size
        (c) Layer 3 primitive at the bounds-edge
    All three must be present."""
    ids = {p["_adversarial_id"] for p in ADVERSARIAL_PARAM_SETS}
    assert any("louver" in i for i in ids), f"missing louver adversarial in {ids}"
    assert any("tpms" in i for i in ids), f"missing tpms adversarial in {ids}"
    assert any("primitive" in i or "prim" in i for i in ids), (
        f"missing primitive adversarial in {ids}"
    )


def test_adversarial_louver_set_is_clustered_at_tip() -> None:
    """Adversarial set (a) must actually push louvers toward the tip
    (cluster_tip ≈ 1.0) at minimum spacing."""
    a = next(p for p in ADVERSARIAL_PARAM_SETS if p["_adversarial_id"].startswith("a_louver"))
    assert a["louver_active"] is True
    assert a["louver_cluster_tip"] >= 0.9
    assert a["louver_spacing_m"] <= 0.006  # minimum or near-minimum


def test_adversarial_tpms_set_uses_minimum_cell_and_rib_crossing_rotation() -> None:
    b = next(p for p in ADVERSARIAL_PARAM_SETS if p["_adversarial_id"].startswith("b_tpms"))
    assert b["tpms_active"] is True
    assert b["tpms_cell_size_m"] <= 0.007  # at or near minimum cell
    # 45° rotation puts through-cuts across both ribs + click region.
    assert 0.5 < b["tpms_rotation_rad"] < 1.1


def test_adversarial_primitive_set_is_at_bounds_edge() -> None:
    c = next(
        p for p in ADVERSARIAL_PARAM_SETS if p["_adversarial_id"].startswith("c_primitive")
    )
    assert c["prim_active"] is True
    # The primitive sits at the boundary of the 5 mm click-clearance lock.
    from fanopt.geometry.spike_0_7a import (
        L_BLADE_M,
        PANEL_TANGENTIAL_OUTER_M,
        RIB_TIP_TAPER_M,
    )
    boundary_x = L_BLADE_M - RIB_TIP_TAPER_M - 0.005
    assert abs(c["prim_x_m"] - boundary_x) < 1e-9
    assert abs(abs(c["prim_y_m"]) - (PANEL_TANGENTIAL_OUTER_M - 0.005)) < 1e-9


# ── evaluate_param_set ──────────────────────────────────────────────────


def test_evaluate_param_set_records_all_four_checks() -> None:
    """Every record must populate the four boolean checks and the hash."""
    params = {"is_adversarial": False, "x": 1}
    rec = evaluate_param_set(
        params,
        generator_fn=_fake_gen_ok,
        manuf_fn=_fake_pass,
        click_check_fn=_fake_pass,
        rib_check_fn=_fake_pass,
    )
    assert isinstance(rec, GeomSanityRecord)
    assert rec.generated is True
    assert rec.manufacturability_passed is True
    assert rec.click_footprint_intact is True
    assert rec.rib_material_preserved is True
    assert rec.passed is True
    assert rec.rejection_reasons == ()
    assert len(rec.params_hash) == 12
    assert rec.is_adversarial is False


def test_evaluate_param_set_accumulates_rejection_reasons() -> None:
    params = {"is_adversarial": True}
    rec = evaluate_param_set(
        params,
        generator_fn=_fake_gen_ok,
        manuf_fn=_fake_fail("thickness out of bounds"),
        click_check_fn=_fake_fail("louver touches click footprint"),
        rib_check_fn=_fake_pass,
    )
    assert rec.passed is False
    assert rec.blocked is True
    assert any("thickness" in r for r in rec.rejection_reasons)
    assert any("louver" in r for r in rec.rejection_reasons)


def test_evaluate_param_set_short_circuits_when_generator_returns_none() -> None:
    rec = evaluate_param_set(
        {"is_adversarial": False},
        generator_fn=_fake_gen_none,
        manuf_fn=_fake_pass,
        click_check_fn=_fake_pass,
        rib_check_fn=_fake_pass,
    )
    assert rec.generated is False
    assert rec.manufacturability_passed is False
    assert rec.click_footprint_intact is False
    assert rec.rib_material_preserved is False
    assert rec.rejection_reasons  # non-empty


def test_evaluate_param_set_catches_generator_exception() -> None:
    def _boom(_p: dict[str, Any]) -> Any:
        raise RuntimeError("synthetic generator failure")

    rec = evaluate_param_set(
        {"is_adversarial": False},
        generator_fn=_boom,
        manuf_fn=_fake_pass,
        click_check_fn=_fake_pass,
        rib_check_fn=_fake_pass,
    )
    assert rec.generated is False
    assert any("synthetic generator failure" in r for r in rec.rejection_reasons)


def test_evaluate_param_set_catches_downstream_exception() -> None:
    def _boom(_p: dict[str, Any], _b: Any) -> tuple[bool, tuple[str, ...]]:
        raise ValueError("synthetic manuf failure")

    rec = evaluate_param_set(
        {"is_adversarial": False},
        generator_fn=_fake_gen_ok,
        manuf_fn=_boom,
        click_check_fn=_fake_pass,
        rib_check_fn=_fake_pass,
    )
    assert rec.generated is True
    assert rec.manufacturability_passed is False
    assert any("synthetic manuf failure" in r for r in rec.rejection_reasons)


# ── analyze_07a ─────────────────────────────────────────────────────────


def _mk_record(
    is_adv: bool, passed: bool, reason: str | None = None
) -> GeomSanityRecord:
    return GeomSanityRecord(
        params_hash=hash_params({"_marker": np.random.random()}),
        is_adversarial=is_adv,
        generated=True,
        manufacturability_passed=passed,
        click_footprint_intact=passed,
        rib_material_preserved=passed,
        rejection_reasons=() if passed else (reason or "failed",),
    )


def test_analyze_07a_passes_when_all_adversarial_blocked_and_random_pass_rate_met() -> None:
    """≥7/10 random pass + all 3 adversarial blocked → passed."""
    records = [_mk_record(False, True) for _ in range(8)] + [_mk_record(False, False, "x")] * 2
    records += [_mk_record(True, False, "adv blocked") for _ in range(3)]
    result = analyze_07a(records)
    assert isinstance(result, Spike07aResult)
    assert result.n_random == 10
    assert result.n_adversarial == 3
    assert result.n_passing == 8
    assert result.adversarial_blocked_count == 3
    assert result.passed is True


def test_analyze_07a_fails_when_one_adversarial_slips_through() -> None:
    """Even with a great random pass rate, ONE adversarial that passes is a
    spike failure."""
    records = [_mk_record(False, True) for _ in range(10)]
    records += [_mk_record(True, False, "blocked") for _ in range(2)]
    records += [_mk_record(True, True)]  # adversarial that passed every check
    result = analyze_07a(records)
    assert result.adversarial_blocked_count == 2
    assert result.passed is False


def test_analyze_07a_passes_via_documented_failures_when_pass_rate_is_low() -> None:
    """If the pass fraction is below 70 % but ≥ 3 failures have documented
    rejection reasons, the spike still passes — the failures are *informative*."""
    records = [
        _mk_record(False, True),
        _mk_record(False, True),
        _mk_record(False, False, "manuf rejected: thickness"),
        _mk_record(False, False, "manuf rejected: spacing"),
        _mk_record(False, False, "manuf rejected: TPMS cell"),
    ]
    records += [_mk_record(True, False, "blocked") for _ in range(3)]
    result = analyze_07a(records)
    assert result.n_passing == 2
    assert result.n_random == 5
    # pass_fraction = 0.4 < 0.7, but documented failures = 3 ≥ gate.
    assert result.passed is True


def test_analyze_07a_fails_when_random_pass_rate_low_and_failures_undocumented() -> None:
    """Pass fraction < 70 % AND < 3 documented failures → spike fails."""
    # 3/10 pass, 7 fail; ONLY 2 of them have rejection reasons (undocumented).
    records: list[GeomSanityRecord] = []
    records.extend(_mk_record(False, True) for _ in range(3))
    # 2 documented failures
    records.append(_mk_record(False, False, "doc1"))
    records.append(_mk_record(False, False, "doc2"))
    # 5 undocumented failures (passed=False but empty reasons)
    for _ in range(5):
        records.append(
            GeomSanityRecord(
                params_hash=hash_params({"_m": np.random.random()}),
                is_adversarial=False,
                generated=True,
                manufacturability_passed=False,
                click_footprint_intact=True,
                rib_material_preserved=True,
                rejection_reasons=(),
            )
        )
    records.extend(_mk_record(True, False, "blocked") for _ in range(3))
    result = analyze_07a(records)
    assert result.n_passing == 3
    assert result.adversarial_blocked_count == 3
    assert RANDOM_DOCUMENTED_FAIL_GATE == 3
    # 2 documented < 3 gate; pass_fraction = 0.3 < 0.7 → fail.
    assert result.passed is False


def test_analyze_07a_rejects_empty_records() -> None:
    with pytest.raises(ValueError):
        analyze_07a([])


def test_analyze_07a_fails_when_no_adversarial_present() -> None:
    """If the run skipped adversarial entirely, ``adv_ok`` is False → fail."""
    records = [_mk_record(False, True) for _ in range(10)]
    result = analyze_07a(records)
    assert result.n_adversarial == 0
    assert result.passed is False


# ── hash_params ──────────────────────────────────────────────────────────


def test_hash_params_is_stable_for_same_dict() -> None:
    p = {"a": 1, "b": 2.5, "c": "hello"}
    assert hash_params(p) == hash_params(dict(p))


def test_hash_params_differs_when_value_changes() -> None:
    assert hash_params({"a": 1}) != hash_params({"a": 2})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
