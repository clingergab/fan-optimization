"""Picklable CFD objective for process-parallel campaign evaluation.

The campaign parallelizes evaluations across **processes**, not threads: gmsh
installs a main-thread-only signal handler at ``initialize()`` and keeps a single
global model, so concurrent meshing in threads raises ``signal only works in main
thread`` and would corrupt state anyway. A process pool gives each worker its own
main thread + isolated gmsh — and ``ProcessPoolExecutor`` pickles the callable to
the workers, so the objective must be an importable, picklable object (this class)
rather than the closure used on the serial path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fanopt.bo.codec import decode
from fanopt.bo.inertia import fan_i_wrist_kgm2
from fanopt.bo.objective import SliceEvalConfig, evaluate_design
from fanopt.bo.structural import panel_tip_deflection_m
from fanopt.utils.ledger import design_hash

__all__ = ["CfdObjective"]


@dataclass(frozen=True)
class CfdObjective:
    """Picklable ``vector -> (J_fan, I_wrist, structural)`` for a process pool.

    Each design gets a stable per-hash workdir under ``out_dir/designs`` so a
    resumed campaign reuses prior CFD output. Attributes are all picklable
    (``Path`` / ``str`` / frozen :class:`SliceEvalConfig`).
    """

    out_dir: Path
    su2_bin: str | None = None
    eval_cfg: SliceEvalConfig = field(default_factory=SliceEvalConfig)

    def __call__(self, vector: np.ndarray) -> tuple[float, float, float]:
        layer1 = decode(vector)
        workdir = self.out_dir / "designs" / design_hash(layer1.to_dict())
        try:
            res = evaluate_design(
                vector,
                workdir,
                su2_bin=self.su2_bin,
                cfg=self.eval_cfg,
                inertia_fn=fan_i_wrist_kgm2,
                structural_fn=panel_tip_deflection_m,
            )
            if res.i_wrist_kgm2 is None or res.structural is None:  # pragma: no cover - injected
                raise RuntimeError("CFD objective expects inertia + structural evaluators")
            return (float(res.j_fan), float(res.i_wrist_kgm2), float(res.structural))
        except Exception as exc:  # fault isolation: a bad design is penalized, not fatal
            # A degenerate geometry (CadQuery/gmsh throws) or a hard SU2 divergence
            # (run_su2 non-zero exit) must not kill a multi-hour campaign. Record it
            # and return a non-finite objective; the orchestrator sanitizes it to a
            # dominated point so the optimizer just avoids that region.
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            nan = float("nan")
            return (nan, nan, nan)
