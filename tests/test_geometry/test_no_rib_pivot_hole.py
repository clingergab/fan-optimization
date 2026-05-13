"""§0 row 25 + §3.1.2 lock: rib SIMP TO carries NO pivot hole under panel-pivot architecture.

Round-7 CRIT-2 established the original gate (scan src/fanopt/geometry/schema.py
for rib_pivot_hole_* parameters). Round-9 HIGH-9 extends the gate to also scan
§9.1 generate_rib + generate_guard_stick code blocks for `.hole(` patterns and
for pivot_hole_dia / pivot_offset parameters in their signatures.

Under the panel-pivot architecture (§0 row 25), the 3 mm pivot pin runs through
the panel at y = 0, NOT through the rib. The rib radial extent under C7 +
Architectural A is [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m —
the rib doesn't even reach the pivot's blade-frame x = 0.008 m position.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_spec_path() -> Path:
    """Spec lives at either docs/report-final.md or repo-root report-final.md."""
    for candidate in (REPO_ROOT / "docs" / "report-final.md", REPO_ROOT / "report-final.md"):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "docs" / "report-final.md"


SPEC_PATH = _resolve_spec_path()


def _read_spec() -> str:
    if not SPEC_PATH.exists():
        pytest.skip(f"spec not found at {SPEC_PATH}")
    return SPEC_PATH.read_text(encoding="utf-8")


def _extract_function_body(spec: str, fn_name: str) -> str | None:
    """Extract the body of a Python function from a fenced code block in the spec.

    Returns the body text (between `def fn_name(...):` and the next `def ` or
    the end of the fenced code block), or None if the function is not found.
    """
    # Look for the function signature, then capture everything up to the next
    # top-level `def ` or the closing ``` of the fenced code block.
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\([^)]*\)\s*:\s*\n"
        r"(.*?)"
        r"(?=^def\s+\w+\s*\(|^```|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(spec)
    return m.group(1) if m else None


def _extract_function_signature(spec: str, fn_name: str) -> str | None:
    """Extract the full signature (across multi-line parameter lists) of a function."""
    pattern = re.compile(
        rf"def\s+{re.escape(fn_name)}\s*\((.*?)\)\s*:",
        re.DOTALL,
    )
    m = pattern.search(spec)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────
# Original Round-7 CRIT-2 check: schema.py has no rib_pivot_hole_* params
# ─────────────────────────────────────────────────────────────────────


def test_no_rib_pivot_hole_in_schema() -> None:
    """Original Round-7 check: no rib_pivot_hole_* params in schema.py."""
    schema = REPO_ROOT / "src" / "fanopt" / "geometry" / "schema.py"
    if not schema.exists():
        pytest.skip("schema.py not yet written; gate vacuously passes (Phase 0 sets it up)")
    content = schema.read_text(encoding="utf-8")
    matches = re.findall(r"\brib_pivot_hole_\w*\b", content)
    assert not matches, (
        f"§3.1.2 lock violation: rib_pivot_hole_* parameter(s) found in schema.py: "
        f"{matches}. Under panel-pivot architecture, the rib carries NO pivot hole — "
        "the pivot lives in the panel."
    )


# ─────────────────────────────────────────────────────────────────────
# Round-9 HIGH-9 extension: scan §9.1 code blocks
# ─────────────────────────────────────────────────────────────────────


def test_no_pivot_hole_in_generate_rib_code_block() -> None:
    """HIGH-9 Round-9 extension: scan §9.1 generate_rib body for .hole( calls."""
    spec = _read_spec()
    body = _extract_function_body(spec, "generate_rib")
    if body is None:
        pytest.skip("generate_rib function block not found in spec")
    hole_calls = re.findall(r"\.hole\s*\(", body)
    assert not hole_calls, (
        f"HIGH-9 Round-9 violation: generate_rib function body contains "
        f"{len(hole_calls)} .hole(...) call(s) — this drills a pivot hole through "
        "the rib, violating the panel-pivot architecture lock. Strip the "
        ".center(pivot_offset, 0).hole(...) chain from the function body."
    )


def test_no_pivot_hole_in_generate_guard_stick_code_block() -> None:
    """HIGH-9 Round-9 extension: scan §9.1 generate_guard_stick body for .hole(."""
    spec = _read_spec()
    body = _extract_function_body(spec, "generate_guard_stick")
    if body is None:
        pytest.skip("generate_guard_stick function block not found in spec")
    hole_calls = re.findall(r"\.hole\s*\(", body)
    assert not hole_calls, (
        f"HIGH-9 Round-9 violation: generate_guard_stick body contains "
        f"{len(hole_calls)} .hole(...) call(s) — guard sticks have no pivot hole "
        "under panel-pivot architecture."
    )


def test_pivot_hole_params_not_in_generate_rib_signature() -> None:
    """HIGH-9 Round-9 extension: generate_rib signature must NOT carry
    pivot_hole_dia / pivot_offset defaults."""
    spec = _read_spec()
    sig = _extract_function_signature(spec, "generate_rib")
    if sig is None:
        pytest.skip("generate_rib signature not found in spec")
    assert "pivot_hole_dia" not in sig, (
        "HIGH-9 Round-9 violation: generate_rib signature still has pivot_hole_dia "
        "parameter — remove it. Panel-pivot architecture means the rib has no pivot hole."
    )
    assert "pivot_offset" not in sig, (
        "HIGH-9 Round-9 violation: generate_rib signature still has pivot_offset "
        "parameter — remove it."
    )


def test_pivot_hole_params_not_in_generate_guard_stick_signature() -> None:
    """HIGH-9 Round-9 extension: generate_guard_stick signature must NOT carry
    pivot_hole_dia / pivot_offset defaults."""
    spec = _read_spec()
    sig = _extract_function_signature(spec, "generate_guard_stick")
    if sig is None:
        pytest.skip("generate_guard_stick signature not found in spec")
    assert "pivot_hole_dia" not in sig, (
        "HIGH-9 Round-9 violation: generate_guard_stick signature still has "
        "pivot_hole_dia parameter — remove it."
    )
    assert "pivot_offset" not in sig, (
        "HIGH-9 Round-9 violation: generate_guard_stick signature still has "
        "pivot_offset parameter — remove it."
    )


if __name__ == "__main__":
    test_no_rib_pivot_hole_in_schema()
    test_no_pivot_hole_in_generate_rib_code_block()
    test_no_pivot_hole_in_generate_guard_stick_code_block()
    test_pivot_hole_params_not_in_generate_rib_signature()
    test_pivot_hole_params_not_in_generate_guard_stick_signature()
    print("OK: no rib pivot holes anywhere.")
