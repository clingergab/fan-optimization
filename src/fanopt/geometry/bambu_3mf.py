"""Write a single-object Bambu Studio project (.3mf) by swapping the mesh into an existing project.

Every printer / filament / process setting comes from a template project (the operator's own, or one
Bambu Studio exports from its system presets, see :func:`resolve_bambu_preset`); only the object mesh, its
name, its plate placement and an explicit set of setting overrides change. The mesh is
written indexed (shared vertices), so a closed 2-manifold stays manifold — unlike STL, which loses
vertex identity and turns two sheets that touch at an edge into a non-manifold edge on re-welding.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

__all__ = [
    "object_model_xml",
    "model_settings_xml",
    "plate_centre_mm",
    "resolve_bambu_preset",
    "write_bambu_project",
]

_OBJECT_PATH = "3D/Objects/object_1.model"
_RELS_PATH = "3D/_rels/3dmodel.model.rels"
_PACKAGE_RELS_PATH = "_rels/.rels"
_SETTINGS_PATH = "Metadata/project_settings.config"
# Everything else in a template (thumbnails, sliced gcode, plate/cut/sequence metadata) describes the
# template's own object and would be stale.
_KEPT = (
    "[Content_Types].xml",
    "3D/3dmodel.model",
    _SETTINGS_PATH,
    "Metadata/model_settings.config",
)
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_P_NS = 'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"'
_NS = (
    'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
    'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
    'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"'
)


def object_model_xml(verts_mm: np.ndarray, faces: np.ndarray) -> str:
    """The ``3D/Objects/object_1.model`` part: one mesh object (id 1), millimetres."""
    v = np.asarray(verts_mm, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    head = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {_NS}>\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n'
        '  <object id="1" p:UUID="00010000-81cb-4c03-9d28-80fed5dfa1dc" type="model">\n   <mesh>\n    <vertices>\n'
    )
    vtx = "".join(f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>\n' for x, y, z in v)
    tri = "".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in f)
    tail = "    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n"
    return head + vtx + "    </vertices>\n    <triangles>\n" + tri + tail


def model_settings_xml(name: str, n_faces: int, lift_mm: float) -> str:
    """``Metadata/model_settings.config`` for one object on plate 1."""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="2">\n'
        f'    <metadata key="name" value="{name}"/>\n    <metadata key="extruder" value="1"/>\n'
        f'    <metadata face_count="{n_faces}"/>\n    <part id="1" subtype="normal_part">\n'
        f'      <metadata key="name" value="{name}"/>\n'
        '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        f'      <metadata key="source_file" value="{name}"/>\n'
        '      <metadata key="source_object_id" value="0"/>\n      <metadata key="source_volume_id" value="0"/>\n'
        f'      <mesh_stat face_count="{n_faces}" edges_fixed="0" degenerate_facets="0" facets_removed="0" '
        'facets_reversed="0" backwards_edges="0"/>\n    </part>\n  </object>\n'
        '  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n'
        '    <metadata key="locked" value="false"/>\n    <model_instance>\n'
        '      <metadata key="object_id" value="2"/>\n      <metadata key="instance_id" value="0"/>\n'
        '      <metadata key="identify_id" value="1"/>\n    </model_instance>\n  </plate>\n'
        '  <assemble>\n   <assemble_item object_id="2" instance_id="0" '
        f'transform="1 0 0 0 1 0 0 0 1 0 0 {lift_mm:.6f}" offset="0 0 0" />\n  </assemble>\n</config>\n'
    )


def resolve_bambu_preset(profile_dir: Path, kind: str, name: str) -> dict[str, object]:
    """One Bambu Studio system preset with its ``inherits`` chain merged (child keys win), standalone."""
    data = json.loads((profile_dir / kind / f"{name}.json").read_text(encoding="utf-8"))
    parent = data.pop("inherits", None)
    merged = resolve_bambu_preset(profile_dir, kind, str(parent)) if parent else {}
    merged.update(data)
    merged["name"] = name
    return merged


def _bbox(points: list[str]) -> tuple[float, float, float, float]:
    xy = np.array([[float(c) for c in p.split("x")] for p in points])
    return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()


def plate_centre_mm(cfg: dict) -> tuple[float, float]:
    """Centre of the bed region every nozzle can reach (``printable_area`` ∩ each ``extruder_printable_area``)."""
    regions = [list(cfg["printable_area"])]
    regions += [str(a).split(",") for a in cfg.get("extruder_printable_area", []) if a]
    boxes = np.array([_bbox(r) for r in regions])
    x0, y0 = boxes[:, :2].max(axis=0)
    x1, y1 = boxes[:, 2:].min(axis=0)
    return float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)


def _root_model(template_root: str, plate_xy_mm: tuple[float, float], lift_mm: float) -> str:
    resources = (
        '<resources>\n  <object id="2" p:UUID="00000002-61cb-4c03-9d28-80fed5dfa1dc" type="model">\n'
        f'   <components>\n    <component p:path="/{_OBJECT_PATH}" objectid="1" '
        'p:UUID="00010000-b206-40ff-9872-83e8017abed1" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
        "   </components>\n  </object>\n </resources>"
    )
    build = (
        '<build>\n  <item objectid="2" p:UUID="00000002-b1ec-4553-aec9-835e5b724bb4" '
        f'transform="1 0 0 0 1 0 0 0 1 {plate_xy_mm[0]:.6f} {plate_xy_mm[1]:.6f} {lift_mm:.6f}" printable="1"/>\n </build>'
    )
    s = template_root
    # The root <model> tag must declare the production namespace the component path uses.
    if "xmlns:p=" not in s.split(">", 2)[1]:
        s = s.replace("<model ", f"<model {_P_NS} ", 1)
    s = re.sub(r' *<metadata name="Thumbnail_[^"]*">[^<]*</metadata>\n?', "", s)
    # A sliced project (.gcode.3mf) saves the empty elements self-closed: <resources/>, <build/>.
    s = re.sub(r"<resources\s*/>|<resources>.*</resources>", lambda _m: resources, s, flags=re.S)
    return re.sub(r"<build\b[^>]*/>|<build\b[^>]*>.*</build>", lambda _m: build, s, flags=re.S)


def _rels_xml(target: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="{_RELS_NS}">\n'
        f' <Relationship Target="{target}" Id="rel-1" Type="{_MODEL_REL}"/>\n</Relationships>\n'
    )


def write_bambu_project(
    template: Path,
    out: Path,
    verts_mm: np.ndarray,
    faces: np.ndarray,
    *,
    name: str,
    settings: dict[str, object],
    plate_xy_mm: tuple[float, float] | None = None,
) -> Path:
    """Copy ``template`` to ``out`` with the mesh replaced and ``settings`` applied to the project config.

    The mesh is centred on its bounding box, lifted so its lowest point sits on the bed and placed at
    ``plate_xy_mm`` (default: the centre of the bed every nozzle can reach). Only the root model, the
    content types and the two configs are taken from the template; the rest is stale and dropped.
    """
    v = np.asarray(verts_mm, dtype=float)
    v = v - (v.max(axis=0) + v.min(axis=0)) / 2.0
    lift = float(-v[:, 2].min())
    with zipfile.ZipFile(template) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        cfg = json.loads(zin.read(_SETTINGS_PATH))
        cfg.update(settings)
        xy = plate_centre_mm(cfg) if plate_xy_mm is None else plate_xy_mm
        for item in zin.infolist():
            n = item.filename
            if n not in _KEPT:
                continue
            data = zin.read(n)
            if n == "3D/3dmodel.model":
                data = _root_model(data.decode("utf-8"), xy, lift).encode("utf-8")
            elif n == "Metadata/model_settings.config":
                data = model_settings_xml(name, len(faces), lift).encode("utf-8")
            elif n == _SETTINGS_PATH:
                data = json.dumps(cfg, indent=4).encode("utf-8")
            z.writestr(item, data)
        z.writestr(_PACKAGE_RELS_PATH, _rels_xml("/3D/3dmodel.model"))
        z.writestr(_RELS_PATH, _rels_xml(f"/{_OBJECT_PATH}"))
        z.writestr(_OBJECT_PATH, object_model_xml(v, faces))
    return out
