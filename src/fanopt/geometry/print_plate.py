"""Lay several copies of one single-object Bambu project on a plate, with brim only at a few points.

A tall blade printed on its edge needs some brim to stay down, but a full brim along its 227 mm edge is
material to peel off every blade. Bambu Studio's brim *ears* put a small round brim only where asked;
the points are stored in ``Metadata/brim_ear_points.txt`` (object coordinates, one line per object:
``x y z radius 0`` per ear, all at the object's lowest z). Works on projects written by
:func:`fanopt.geometry.bambu_3mf.write_bambu_project` (one object, one plate).
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

__all__ = [
    "object_vertices_mm",
    "footprint_ear_points_mm",
    "row_positions_mm",
    "write_plate",
]

_OBJECT_PATH = "3D/Objects/object_1.model"
_ROOT_PATH = "3D/3dmodel.model"
_MODEL_SETTINGS_PATH = "Metadata/model_settings.config"
_SETTINGS_PATH = "Metadata/project_settings.config"
_EARS_PATH = "Metadata/brim_ear_points.txt"
_VERTEX = re.compile(rb'<vertex x="([-\d.eE+]+)" y="([-\d.eE+]+)" z="([-\d.eE+]+)"')


def object_vertices_mm(project: Path) -> np.ndarray:
    """The project's mesh vertices in object coordinates (mm), ``(n, 3)``."""
    with zipfile.ZipFile(project) as z:
        data = z.read(_OBJECT_PATH)
    return np.array(_VERTEX.findall(data), dtype=float)


def footprint_ear_points_mm(
    verts_mm: np.ndarray, n_ears: int, *, inset_mm: float, contact_mm: float = 0.2
) -> np.ndarray:
    """``n_ears`` points on the part's bed contact, spread evenly along its long (x) side, ends inset.

    The bed contact is every vertex within ``contact_mm`` of the lowest one. Each point is centred
    across the contact near its x and sits at the lowest z (the slicer drops ears above it).
    """
    v = np.asarray(verts_mm, dtype=float)
    z0 = float(v[:, 2].min())
    foot = v[v[:, 2] <= z0 + contact_mm]
    x0, x1 = float(foot[:, 0].min()) + inset_mm, float(foot[:, 0].max()) - inset_mm
    xs = np.linspace(x0, x1, n_ears) if n_ears > 1 else np.array([(x0 + x1) / 2.0])
    half = max((x1 - x0) / (4.0 * max(n_ears - 1, 1)), 1.0)
    ys = []
    for x in xs:
        dx = np.abs(foot[:, 0] - x)
        reach = max(
            half, float(np.sort(dx)[min(8, len(dx)) - 1])
        )  # a coarse mesh may have no vertex near x
        near = foot[dx <= reach, 1]
        ys.append(float(near.min() + near.max()) / 2.0)
    return np.column_stack([xs, ys, np.full(len(xs), z0)])


def row_positions_mm(
    n: int, pitch_mm: float, centre_xy_mm: tuple[float, float]
) -> list[tuple[float, float]]:
    """``n`` plate positions in one column of parallel rows ``pitch_mm`` apart, centred on the plate."""
    cx, cy = centre_xy_mm
    return [(cx, cy + (k - (n - 1) / 2.0) * pitch_mm) for k in range(n)]


def _ears_txt(points_mm: np.ndarray, radius_mm: float) -> str:
    ears = "".join(f"{x:f} {y:f} {z:f} {radius_mm:f} 0 " for x, y, z in points_mm)
    return f"brim_points_format_version=1\nobject_id=1|{ears}\n"


def _root_with_items(root: str, positions_mm: list[tuple[float, float]]) -> str:
    item = re.search(r"<item\b[^>]*/>", root)
    if item is None:
        raise ValueError("project has no build item to copy")
    tf = re.search(r'transform="([^"]+)"', item.group(0)).group(1).split()
    items = []
    for k, (x, y) in enumerate(positions_mm):
        t = " ".join([*tf[:9], f"{x:.6f}", f"{y:.6f}", tf[11]])
        tag = re.sub(r'transform="[^"]+"', f'transform="{t}"', item.group(0))
        items.append(re.sub(r'p:UUID="[^"]+"', f'p:UUID="00000002-b1ec-4553-aec9-{k:012d}"', tag))
    return root.replace(item.group(0), "\n  ".join(items))


def _model_settings_with_instances(cfg: str, n: int) -> str:
    inst = re.search(r"<model_instance>.*?</model_instance>", cfg, flags=re.S)
    asm = re.search(r"<assemble_item\b[^>]*/>", cfg)
    if inst is None or asm is None:
        raise ValueError("model settings have no instance to copy")
    insts, asms = [], []
    for k in range(n):
        i = re.sub(r'(key="instance_id" value=")\d+', rf"\g<1>{k}", inst.group(0))
        insts.append(re.sub(r'(key="identify_id" value=")\d+', rf"\g<1>{k + 1}", i))
        asms.append(re.sub(r'instance_id="\d+"', f'instance_id="{k}"', asm.group(0)))
    cfg = cfg.replace(inst.group(0), "\n    ".join(insts))
    return cfg.replace(asm.group(0), "\n   ".join(asms))


def write_plate(
    src: Path,
    out: Path,
    positions_mm: list[tuple[float, float]],
    *,
    ear_points_mm: np.ndarray,
    ear_radius_mm: float,
) -> Path:
    """Copy the single-object project ``src`` to ``out`` with one copy per plate position and brim ears.

    Every copy keeps the original orientation and height; ``brim_type`` becomes ``brim_ears`` so the
    slicer prints brim only at ``ear_points_mm`` (object coordinates, shared by every copy).
    """
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in zin.infolist():
            n = item.filename
            if n == _EARS_PATH:
                continue
            data = zin.read(n)
            if n == _ROOT_PATH:
                data = _root_with_items(data.decode("utf-8"), positions_mm).encode("utf-8")
            elif n == _MODEL_SETTINGS_PATH:
                cfg = _model_settings_with_instances(data.decode("utf-8"), len(positions_mm))
                data = cfg.encode("utf-8")
            elif n == _SETTINGS_PATH:
                settings = json.loads(data)
                settings["brim_type"] = "brim_ears"
                data = json.dumps(settings, indent=4).encode("utf-8")
            z.writestr(item, data)
        z.writestr(_EARS_PATH, _ears_txt(np.asarray(ear_points_mm, dtype=float), ear_radius_mm))
    return out
