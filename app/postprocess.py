"""Mesh clean-up, orientation, scaling and export helpers (pure trimesh/numpy)."""
from __future__ import annotations

import numpy as np
import trimesh

FORMATS = {
    "stl": ("model/stl", "stl"),
    "obj": ("text/plain", "obj"),
    "glb": ("model/gltf-binary", "glb"),
    "ply": ("application/octet-stream", "ply"),
}


def load_raw(path) -> trimesh.Trimesh:
    return trimesh.load(str(path), file_type="ply", process=False, force="mesh")


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop floating fragments; keep the part with the most faces."""
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda p: len(p.faces))


def _carry_colors(src: trimesh.Trimesh, dst: trimesh.Trimesh) -> None:
    """Copy per-vertex colours from src onto dst (nearest vertex); grey if src has none."""
    from scipy.spatial import cKDTree

    try:
        cols = np.asarray(src.visual.vertex_colors)
        _, idx = cKDTree(src.vertices).query(dst.vertices)
        dst.visual = trimesh.visual.ColorVisuals(dst, vertex_colors=cols[idx])
    except Exception:
        dst.visual = trimesh.visual.ColorVisuals(dst, vertex_colors=np.tile([200, 200, 200, 255], (len(dst.vertices), 1)))


def mirror_x(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = m.copy()
    m.vertices = m.vertices * np.array([-1.0, 1.0, 1.0])
    m.invert()                      # mirroring flips the winding; flip it back
    return m


def decimate(m: trimesh.Trimesh, pct: int) -> trimesh.Trimesh:
    """Reduce to roughly pct % of the triangles (quadric decimation). No-op if unavailable."""
    if pct >= 99:
        return m
    target = max(500, int(len(m.faces) * pct / 100))
    try:
        d = m.simplify_quadric_decimation(face_count=target)
    except Exception:
        return m
    if len(d.faces) == 0:
        return m
    _carry_colors(m, d)
    return d


def cut_base(m: trimesh.Trimesh, pct: float) -> trimesh.Trimesh:
    """Slice off the bottom pct % of the height and cap it flat (gives a flat, printable base)."""
    if pct <= 0:
        return m
    lo, hi = m.bounds
    y = lo[1] + (hi[1] - lo[1]) * pct / 100.0
    try:
        c = m.slice_plane([0, y, 0], [0, 1, 0], cap=True)
    except Exception:
        return m
    if c is None or len(c.faces) == 0:
        return m
    _carry_colors(m, c)
    return c


def add_plinth(m: trimesh.Trimesh, thickness_mm: float, up_axis: int, margin: float = 0.12) -> trimesh.Trimesh:
    """Add a flat box base under a model that is already placed in millimetres (resting on 0)."""
    if thickness_mm <= 0:
        return m
    lo, hi = m.bounds
    size = hi - lo
    side = [a for a in range(3) if a != up_axis]
    for a in side:
        size[a] *= 1 + margin
    size[up_axis] = thickness_mm
    box = trimesh.creation.box(extents=size)
    centre = (lo + hi) / 2
    move = centre.copy()
    move[up_axis] = lo[up_axis] - thickness_mm / 2 + 0.02 * thickness_mm      # slight overlap -> fuses in slicer
    box.apply_translation(move)
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=np.tile([190, 190, 195, 255], (len(box.vertices), 1)))
    base = m.copy()
    _carry_colors(m, base)
    out = trimesh.util.concatenate([box, base])
    drop = np.zeros(3)
    drop[up_axis] = -out.bounds[0][up_axis]
    out.apply_translation(drop)
    return out


def build(raw: trimesh.Trimesh, clean: bool = True, smooth: int = 0, rx: int = 0, ry: int = 0, rz: int = 0,
          mirror: bool = False, decimate_pct: int = 100, cut: float = 0) -> trimesh.Trimesh:
    """Apply the non-destructive edits. Returned mesh is in the Y-up 'preview' frame."""
    m = raw.copy()
    if clean:
        m = keep_largest_component(m)
    if mirror:
        m = mirror_x(m)
    for ang, axis in ((rx, [1, 0, 0]), (ry, [0, 1, 0]), (rz, [0, 0, 1])):
        if ang % 360:
            m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ang), axis))
    m = decimate(m, int(decimate_pct))
    if smooth > 0:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=int(smooth))
    m = cut_base(m, float(cut))
    return m


def place(mesh: trimesh.Trimesh, height_mm: float, z_up: bool, lay_flat: bool = False) -> trimesh.Trimesh:
    """Scale so the model is `height_mm` tall, centre it, and rest it on the ground plane.

    lay_flat: keep the front facing up (+Z) and lie on its back - for plaques/reliefs.
              Then `height_mm` is the longest side in the XY plane.
    """
    m = mesh.copy()
    if lay_flat:
        m.apply_scale(height_mm / max(m.extents[0], m.extents[1], 1e-9))
        lo, hi = m.bounds
        m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        return m
    if z_up:  # Y-up -> Z-up (slicer convention)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    up = 2 if z_up else 1
    ext = m.extents[up]
    if ext > 1e-9:
        m.apply_scale(height_mm / ext)
    lo, hi = m.bounds
    shift = np.array([-(lo[0] + hi[0]) / 2, 0, 0], dtype=float)
    shift[up] = -lo[up]
    other = 1 if z_up else 2
    shift[other] = -(lo[other] + hi[other]) / 2
    m.apply_translation(shift)
    return m


def stats(mesh: trimesh.Trimesh) -> dict:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "size": [round(float(v), 2) for v in mesh.extents],
    }


def make_solid(mesh: trimesh.Trimesh, res: int = 280) -> trimesh.Trimesh:
    """Rebuild a leaky mesh as a closed solid by voxelising and re-meshing it.

    Slower and slightly softer than the original surface, but always watertight
    (as long as the shape encloses some volume). Vertex colours are carried over.
    """
    from scipy.spatial import cKDTree

    pitch = float(max(mesh.extents)) / res
    vox = mesh.voxelized(pitch).fill()
    solid = vox.marching_cubes
    solid.update_faces(solid.nondegenerate_faces())
    solid.remove_unreferenced_vertices()
    trimesh.smoothing.filter_taubin(solid, iterations=6)
    try:
        _, idx = cKDTree(mesh.vertices).query(solid.vertices)
        solid.visual.vertex_colors = mesh.visual.vertex_colors[idx]
    except Exception:
        pass
    trimesh.repair.fix_normals(solid)
    return solid


def repair_for_print(m: trimesh.Trimesh) -> tuple[trimesh.Trimesh, str]:
    """Return (mesh, how) where how is 'already-closed' | 'holes-filled' | 'voxel-remesh' | 'failed'."""
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    if m.is_watertight:
        return m, "already-closed"
    trimesh.repair.fill_holes(m)
    if m.is_watertight:
        return m, "holes-filled"
    try:
        s = make_solid(m)
        if s.is_watertight:
            return s, "voxel-remesh"
    except Exception:
        pass
    return m, "failed"


def describe(m: trimesh.Trimesh) -> dict:
    """Measurements for the Info panel (mesh must already be in millimetres)."""
    d = stats(m)
    d["area_cm2"] = round(float(m.area) / 100.0, 2)
    d["volume_cm3"] = round(abs(float(m.volume)) / 1000.0, 2) if m.is_watertight else None
    return d


def export_bytes(mesh: trimesh.Trimesh, fmt: str, height_mm: float = 100.0,
                 lay_flat: bool = False, plinth_mm: float = 0.0) -> tuple[bytes, str, str, dict]:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    flat = lay_flat and fmt == "stl"
    m = place(mesh, height_mm, z_up=(fmt == "stl"), lay_flat=flat)
    if plinth_mm > 0 and not flat:
        m = add_plinth(m, plinth_mm, up_axis=2 if fmt == "stl" else 1)
    repair = None
    if fmt == "stl":
        m, repair = repair_for_print(m)
        if repair == "voxel-remesh":  # re-seat on the bed after remeshing
            m.apply_translation([0, 0, -m.bounds[0][2]])
    _ = m.vertex_normals                      # cache normals so GLB/OBJ/PLY exports carry them
    data = m.export(file_type=fmt)
    if isinstance(data, str):
        data = data.encode()
    mime, ext = FORMATS[fmt]
    st = describe(m)
    st["repair"] = repair
    return data, mime, ext, st
