"""Post-hoc analysis of a blade-campaign ledger (pooled across shards / folders).

Pure/numpy-only (no botorch) so it runs anywhere. Reads the JSONL shards a campaign wrote,
dedups by ``design_hash``, and reports objective stats, the optimization progression (running-best
J_fan in evaluation order — the signal for "is it actually learning or random?"), the Pareto
front, and the best designs. Also surfaces how the data is split across folders/shards, which is
how we detect the Drive duplicate-folder coordination failure.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fanopt.bo.blade_codec import SEARCH_SPACE, decode

__all__ = [
    "load_rows",
    "pareto_indices",
    "campaign_report",
    "find_shards",
    "shape_summary",
    "panel_shape_summary",
    "classify_panel",
    "top_designs_shapes",
    "failed_designs",
    "shape_evolution",
    "session_trajectories",
]

# 5 rib-bow knots run hub->tip; bucket the wave's peak location into a coarse surface "type"
_PEAK_BUCKET = {0: "hub", 1: "hub", 2: "mid", 3: "tip", 4: "tip"}

# Column indices of the panel mean-surface offset knobs in the BO vector (the 4x3 aero grid).
_PANEL_Z_IDX = [i for i, v in enumerate(SEARCH_SPACE) if v.name.startswith("panel_z")]
# Below this peak-to-anchor offset the panel is aerodynamically flat (sub-50-micron ripple).
_PANEL_FLAT_EPS_M = 5.0e-5


def _reversals(profile: np.ndarray) -> int:
    """Interior extrema of a 1-D profile = sign reversals of its slope (tiny steps ignored).

    0 → monotone tilt, 1 → single dish/hump, >=2 → alternating (pleated). Counting slope
    reversals (not centered sign changes) so one symmetric hump reads as one feature, not two.
    """
    d = np.diff(np.asarray(profile, dtype=float))
    amp = float(np.max(profile) - np.min(profile))
    d = d[np.abs(d) > 0.15 * amp] if amp > 0 else np.array([])
    if len(d) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(d)) != 0))


def classify_panel(panel_offsets_m) -> str:
    """Name the panel aero-surface *shape type* from its 4x3 offset grid.

    ``flat`` (offsets below the aero-flat floor), else by the interior extrema of the dominant
    (radial base->tip or chordwise) profile: ``camber`` (<=1 — a monotone tilt or single
    dish/hump) or ``zigzag`` (>=2 — pleated / multi-hump / alternating). The panel analogue of
    the rib-bow ``interp`` label — the thing the free displacement grid exists to discover.
    """
    g = np.asarray(panel_offsets_m, dtype=float)
    if g.size == 0 or np.max(np.abs(g)) < _PANEL_FLAT_EPS_M:
        return "flat"
    return (
        "zigzag" if max(_reversals(g.mean(axis=1)), _reversals(g.mean(axis=0))) >= 2 else "camber"
    )


def load_rows(shard_files: list[str | Path]) -> list[dict]:
    """Parse every JSONL row from the given shard files (skips blank / torn lines)."""
    rows: list[dict] = []
    for f in shard_files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pareto_indices(objectives: np.ndarray) -> list[int]:
    """Indices of the non-dominated rows. ``objectives`` columns = (J_fan↑, mass↓, deflection↓)."""
    if len(objectives) == 0:
        return []
    m = objectives.copy()
    m[:, 1:] = -m[:, 1:]  # to a pure-maximization frame
    keep: list[int] = []
    for i in range(len(m)):
        dominated = np.any(np.all(m >= m[i], axis=1) & np.any(m > m[i], axis=1))
        if not dominated:
            keep.append(i)
    return keep


def campaign_report(shard_files: list[str | Path], *, top_k: int = 10) -> dict:
    """Pool + dedup the shards and summarize the campaign (see module docstring)."""
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash", id(r)), r)
    uniq = list(by_hash.values())
    finite = [
        r for r in uniq if isinstance(r.get("j_fan"), (int | float)) and np.isfinite(r["j_fan"])
    ]

    report: dict = {
        "total_rows": len(rows),
        "unique_designs": len(uniq),
        "finite": len(finite),
        "failed_nan": len(uniq) - len(finite),
        "duplicate_rows": len(rows) - len(uniq),
        "sources": {
            s: sum(1 for r in finite if r.get("source") == s)
            for s in sorted({r.get("source") for r in finite})
        },
    }
    if not finite:
        return report

    j = np.array([r["j_fan"] for r in finite])
    mass = np.array([r["mass_kg"] for r in finite])
    defl = np.array([r.get("deflection_m", float("nan")) for r in finite])
    report["j_fan"] = {"min": float(j.min()), "mean": float(j.mean()), "max": float(j.max())}
    report["mass_g"] = {
        "min": float(mass.min() * 1e3),
        "mean": float(mass.mean() * 1e3),
        "max": float(mass.max() * 1e3),
    }

    # PROGRESSION — the honest "is it learning?" signals, not just the (monotone) running-best.
    order = sorted(range(len(finite)), key=lambda i: finite[i].get("timestamp_iso", ""))
    j_by_time = [float(j[i]) for i in order]
    running_best, best, improvements = [], -np.inf, 0
    for v in j_by_time:
        if v > best:
            best, improvements = v, improvements + 1
        running_best.append(best)
    win = max(5, len(j_by_time) // 15)
    rolling_mean = [
        float(np.mean(j_by_time[max(0, i - win + 1) : i + 1])) for i in range(len(j_by_time))
    ]
    n = len(j_by_time)
    quartile_means = (
        [float(np.mean(j_by_time[k * n // 4 : (k + 1) * n // 4])) for k in range(4)]
        if n >= 4
        else []
    )
    # per-session trajectories (fragmentation: each session learned only on its OWN history)
    by_session: dict[str, dict] = {}
    for r in finite:
        by_session.setdefault(r.get("session_id", "?"), []).append(r)
    session_stats: dict[str, dict] = {}
    for s, rs in by_session.items():
        rs = sorted(rs, key=lambda r: r.get("timestamp_iso", ""))
        seq = [float(r["j_fan"]) for r in rs]
        h = len(seq) // 2
        session_stats[s] = {
            "n": len(seq),
            "j_series": seq,
            "first_half_mean": float(np.mean(seq[:h])) if h else None,
            "second_half_mean": float(np.mean(seq[h:])) if h else None,
            "best": max(seq),
        }
    report["progression"] = {
        "j_by_time": j_by_time,
        "running_best": running_best,
        "rolling_mean": rolling_mean,
        "rolling_window": win,
        "quartile_means": quartile_means,  # mean J_fan per quarter of the run — rising = learning
        "n_new_bests": improvements,
        "by_session": session_stats,
    }
    # BO vs DoE by MEAN (learning = consistently better proposals), not just the lucky best
    sob = [r["j_fan"] for r in finite if r.get("source") == "sobol"]
    bo = [r["j_fan"] for r in finite if r.get("source") == "bo"]
    report["sobol_best"] = max(sob) if sob else None
    report["bo_best"] = max(bo) if bo else None
    report["sobol_mean"] = float(np.mean(sob)) if sob else None
    report["bo_mean"] = float(np.mean(bo)) if bo else None

    obj = np.column_stack([j, mass, defl])
    pf = pareto_indices(obj)
    report["pareto_count"] = len(pf)
    report["pareto"] = sorted(
        (
            {
                "j_fan": float(j[i]),
                "mass_g": float(mass[i] * 1e3),
                "deflection_mm": float(defl[i] * 1e3),
                "blade_count": finite[i].get("blade_count"),
                "design_hash": finite[i].get("design_hash", "")[:8],
            }
            for i in pf
        ),
        key=lambda d: -d["j_fan"],
    )
    report["top_by_j_fan"] = sorted(
        (
            {
                "j_fan": float(j[i]),
                "mass_g": float(mass[i] * 1e3),
                "source": finite[i].get("source"),
                "design_hash": finite[i].get("design_hash", "")[:8],
            }
            for i in range(len(finite))
        ),
        key=lambda d: -d["j_fan"],
    )[:top_k]
    # mass-cap-eligible best (C9 <100 g): the top-J_fan designs are often too heavy — the V1 pick
    # must come from designs under the cap, so surface the best ones that actually qualify.
    report["top_under_mass_cap"] = {
        f"{cap}g": sorted(
            (
                {
                    "j_fan": float(j[i]),
                    "mass_g": float(mass[i] * 1e3),
                    "source": finite[i].get("source"),
                    "design_hash": finite[i].get("design_hash", "")[:8],
                }
                for i in range(len(finite))
                if mass[i] * 1e3 <= cap
            ),
            key=lambda d: -d["j_fan"],
        )[:top_k]
        for cap in (100.0,)
    }
    return report


def shape_summary(shard_files: list[str | Path]) -> dict:
    """Decode designs into their SURFACE SHAPE (rib-bow wave) and report the type distribution,
    which types perform best, and whether the run converged onto one type over time.

    Answers "what shapes did we explore, is the pool skewed to one type, and did later designs
    drift toward a particular shape?" — the coverage check for whether a continuation would keep
    exploiting one region vs genuinely exploring. Each design's ``type`` = (wave-peak location
    hub/mid/tip, interp). Requires the geometry codec (``decode``).
    """
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash"), r)
    finite = sorted(
        (
            r
            for r in by_hash.values()
            if isinstance(r.get("j_fan"), (int | float)) and np.isfinite(r["j_fan"])
        ),
        key=lambda r: r.get("timestamp_iso", ""),
    )
    recs: list[dict] = []
    for r in finite:
        try:
            p = decode(np.array(r["vector"], dtype=float)).to_dict()
        except (KeyError, ValueError, TypeError):
            continue
        knots = p.get("rib_bow_knots_m") or []
        if not knots:
            continue
        recs.append(
            {
                "j_fan": float(r["j_fan"]),
                "peak": _PEAK_BUCKET.get(int(np.argmax(knots)), "?"),
                "interp": p.get("rib_bow_interp"),
                "amplitude_mm": (max(knots) - min(knots)) * 1e3,
                "blade_count": p.get("blade_count"),
                "knots_mm": [k * 1e3 for k in knots],
            }
        )
    if not recs:
        return {"n": 0}
    by_peak: dict[str, list[float]] = {}
    for x in recs:
        by_peak.setdefault(x["peak"], []).append(x["j_fan"])
    third = max(1, len(recs) // 3)
    return {
        "n": len(recs),
        "type_counts": dict(Counter(f"{x['peak']}/{x['interp']}" for x in recs)),
        "peak_counts": dict(Counter(x["peak"] for x in recs)),
        "peak_mean_jfan": {k: float(np.mean(v)) for k, v in by_peak.items()},
        "blade_counts": dict(Counter(x["blade_count"] for x in recs)),
        "early_peak_dist": dict(Counter(x["peak"] for x in recs[:third])),
        "late_peak_dist": dict(Counter(x["peak"] for x in recs[-third:])),
        "mean_amplitude_mm": float(np.mean([x["amplitude_mm"] for x in recs])),
        "records": recs,
    }


def panel_shape_summary(shard_files: list[str | Path]) -> dict:
    """Characterize the PANEL aero surface — the 24-of-33 codec dims the rib-bow summary ignores.

    Answers "what panel shapes did it try, and do they matter?" The panel offset is capped by
    containment to ``(t_rib - panel_thickness)/2`` (sub-mm), so the key number is
    ``authority_pct`` — panel amplitude as a fraction of the rib-bow extent. A low value means
    the winning shape is the rib meridian and the panel is a maxed-but-tiny ripple.
    ``at_bound_pct`` (share of the 12 panel knobs pinned to the containment limit) shows whether
    the optimizer WANTED more panel travel than containment allowed.
    """
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash"), r)
    amp_mm: list[float] = []
    ribbow_mm: list[float] = []
    authority: list[float] = []
    at_bound: list[float] = []
    classes: list[str] = []
    for r in by_hash.values():
        j = r.get("j_fan")
        if not (isinstance(j, (int | float)) and np.isfinite(j)):
            continue
        try:
            vec = np.array(r["vector"], dtype=float)
            p = decode(vec).to_dict()
        except (KeyError, ValueError, TypeError):
            continue
        knots = p.get("rib_bow_knots_m") or []
        offs = np.asarray(p.get("panel_offsets_m") or [], dtype=float)
        if not knots or offs.size == 0:
            continue
        a = float(np.max(np.abs(offs)) * 1e3)
        bow = (max(knots) - min(knots)) * 1e3
        amp_mm.append(a)
        ribbow_mm.append(bow)
        # bow is in mm; 1e-3 mm (1 micron) is the "rib is not flat" floor. A one-sided panel
        # amplitude over a two-sided knot span makes this a rough ~order-of-magnitude proxy, not
        # a calibrated ratio — the qualitative "panel << rib" is what it is meant to show.
        authority.append(a / bow if bow > 1e-3 else 0.0)
        at_bound.append(float(np.mean(np.abs(vec[_PANEL_Z_IDX]) > 0.95)))
        classes.append(classify_panel(offs))
    if not amp_mm:
        return {"n": 0}
    return {
        "n": len(amp_mm),
        "panel_amp_mm": {
            "median": float(np.median(amp_mm)),
            "p10": float(np.percentile(amp_mm, 10)),
            "p90": float(np.percentile(amp_mm, 90)),
            "max": float(np.max(amp_mm)),
        },
        "rib_bow_mm_median": float(np.median(ribbow_mm)),
        "authority_pct_median": float(np.median(authority) * 100.0),
        "at_bound_pct_median": float(np.median(at_bound) * 100.0),
        "class_counts": dict(Counter(classes)),
    }


def top_designs_shapes(shard_files: list[str | Path], k: int = 8) -> list[dict]:
    """Decode the top-``k`` designs by J_fan and show their actual surface shape (the rib-bow wave
    knots hub->tip, peak location, interp, blade_count) — 'what do the winners look like?'"""
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash"), r)
    finite = sorted(
        (
            r
            for r in by_hash.values()
            if isinstance(r.get("j_fan"), (int | float)) and np.isfinite(r["j_fan"])
        ),
        key=lambda r: -r["j_fan"],
    )
    out: list[dict] = []
    for r in finite[:k]:
        try:
            p = decode(np.array(r["vector"], dtype=float)).to_dict()
        except (KeyError, ValueError, TypeError):
            continue
        knots = p.get("rib_bow_knots_m") or []
        offs = np.asarray(p.get("panel_offsets_m") or [], dtype=float)
        out.append(
            {
                "j_fan": float(r["j_fan"]),
                "mass_g": float(r["mass_kg"] * 1e3),
                "peak": _PEAK_BUCKET.get(int(np.argmax(knots)), "?") if knots else "?",
                "interp": p.get("rib_bow_interp"),
                "blade_count": p.get("blade_count"),
                "knots_mm": [round(kk * 1e3, 1) for kk in knots],
                "panel_class": classify_panel(offs) if offs.size else "?",
                "panel_amp_mm": round(float(np.max(np.abs(offs)) * 1e3), 2) if offs.size else 0.0,
                "hash": r.get("design_hash", "")[:8],
            }
        )
    return out


def failed_designs(shard_files: list[str | Path]) -> list[dict]:
    """The designs that FAILED (NaN objective — infeasible geometry or a diverged CFD run), with
    their decoded shape and full vector, so you can see which failed and re-render them."""
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash"), r)
    out: list[dict] = []
    for r in by_hash.values():
        j = r.get("j_fan")
        if isinstance(j, (int | float)) and np.isfinite(j):
            continue  # succeeded
        try:
            p = decode(np.array(r["vector"], dtype=float)).to_dict()
        except (KeyError, ValueError, TypeError):
            p = {}
        knots = p.get("rib_bow_knots_m") or []
        out.append(
            {
                "hash": r.get("design_hash", "")[:8],
                "source": r.get("source"),
                "peak": _PEAK_BUCKET.get(int(np.argmax(knots)), "?") if knots else "?",
                "blade_count": p.get("blade_count"),
                "knots_mm": [round(k * 1e3, 1) for k in knots],
                "vector": r.get("vector"),
            }
        )
    return out


def shape_evolution(shard_files: list[str | Path], n_samples: int = 12) -> list[dict]:
    """Designs sampled evenly across EVALUATION ORDER, each with its rib-bow wave — so you can SEE
    how the surface shape itself evolved over the run (early → late), not just J_fan.

    Returns records ordered from first to last: ``eval_index`` (position in time order),
    ``eval_frac`` (0=first, 1=last), ``j_fan``, ``knots_mm`` (wave height hub→tip), ``peak``.
    """
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash"), r)
    finite = sorted(
        (
            r
            for r in by_hash.values()
            if isinstance(r.get("j_fan"), (int | float)) and np.isfinite(r["j_fan"])
        ),
        key=lambda r: r.get("timestamp_iso", ""),
    )
    if not finite:
        return []
    idxs = np.unique(np.linspace(0, len(finite) - 1, min(n_samples, len(finite))).astype(int))
    out: list[dict] = []
    for i in idxs:
        r = finite[int(i)]
        try:
            knots = (
                decode(np.array(r["vector"], dtype=float)).to_dict().get("rib_bow_knots_m") or []
            )
        except (KeyError, ValueError, TypeError):
            continue
        out.append(
            {
                "eval_index": int(i),
                "eval_frac": float(i) / (len(finite) - 1) if len(finite) > 1 else 0.0,
                "j_fan": float(r["j_fan"]),
                "knots_mm": [round(k * 1e3, 2) for k in knots],
                "peak": _PEAK_BUCKET.get(int(np.argmax(knots)), "?") if knots else "?",
            }
        )
    return out


def session_trajectories(shard_files: list[str | Path]) -> dict[str, list[dict]]:
    """Per session, the designs in EVALUATION (time) order with J_fan, mass and decoded shape — the
    per-point detail an interactive progression plot shows on hover.

    ``{session_id: [{eval, j_fan, mass_g, peak, interp, blade_count, hash}, ...]}`` (each list in
    that session's own time order).
    """
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash"), r)
    by_sess: dict[str, list[dict]] = {}
    for r in by_hash.values():
        j = r.get("j_fan")
        if not (isinstance(j, (int | float)) and np.isfinite(j)):
            continue
        by_sess.setdefault(r.get("session_id", "?"), []).append(r)
    out: dict[str, list[dict]] = {}
    for s, rs in by_sess.items():
        rs = sorted(rs, key=lambda r: r.get("timestamp_iso", ""))
        pts: list[dict] = []
        for idx, r in enumerate(rs):
            try:
                p = decode(np.array(r["vector"], dtype=float)).to_dict()
            except (KeyError, ValueError, TypeError):
                p = {}
            knots = p.get("rib_bow_knots_m") or []
            pts.append(
                {
                    "eval": idx + 1,
                    "j_fan": float(r["j_fan"]),
                    "mass_g": float(r["mass_kg"] * 1e3),
                    "peak": _PEAK_BUCKET.get(int(np.argmax(knots)), "?") if knots else "?",
                    "interp": p.get("rib_bow_interp"),
                    "blade_count": p.get("blade_count"),
                    "hash": r.get("design_hash", "")[:8],
                }
            )
        out[s] = pts
    return out


def find_shards(root: str | Path, pattern: str = "blade_campaign*") -> dict[str, list[str]]:
    """Map each campaign folder under ``root`` to its ledger shards — exposes duplicate folders."""
    out: dict[str, list[str]] = {}
    for d in sorted(glob.glob(str(Path(root) / pattern))):
        shards = sorted(glob.glob(str(Path(d) / "evaluations_*.jsonl")))
        if shards:
            out[d] = shards
    return out
