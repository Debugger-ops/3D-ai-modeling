"""Silhouette 'inflate' engine: line art / logos / sketches -> solid, watertight, printable model.

No AI. The outline of the drawing is cut out (background = light pixels connected to the border),
puffed into a smooth dome using a distance transform, and the dark pen lines are engraved
into the surface. A marching-cubes pass over the resulting volume guarantees a closed mesh.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def foreground_mask(img: Image.Image, tol: float = 40.0, close_px: int = 3) -> np.ndarray:
    """True where the subject is. Uses real transparency if present, else the border colour."""
    rgba = np.asarray(img.convert("RGBA")).astype(np.float32)
    a, rgb = rgba[..., 3], rgba[..., :3]
    if (a < 128).mean() > 0.01:
        fg = a >= 128
    else:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg_col = np.median(border, axis=0)
        fg = np.abs(rgb - bg_col).max(axis=-1) > tol
    sealed = ndi.binary_closing(fg, structure=_disk(close_px)) if close_px > 0 else fg
    lab, _ = ndi.label(~sealed)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    mask = ~np.isin(lab, edge)        # everything not reachable from the border
    mask |= fg
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n > 1:                         # keep the biggest blob only
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    return mask


def generate(image: Image.Image, resolution: int = 320, puff: float = 0.30, detail: float = 0.5,
             flat_back: bool = True, progress=lambda *_: None) -> trimesh.Trimesh:
    """puff: 0.05..0.6 dome thickness (fraction of half the longest side). detail: 0..1 engraving depth."""
    progress("Cutting out silhouette", 0.1)
    img = image.convert("RGBA")
    scale = resolution / max(img.size)
    img = img.resize((max(8, round(img.width * scale)), max(8, round(img.height * scale))), Image.LANCZOS)
    pad = 8
    mask = np.pad(foreground_mask(img), pad)
    rgb = np.pad(np.asarray(img.convert("RGB")), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    lum = rgb.astype(np.float32).mean(-1) / 255.0
    side = max(mask.shape)

    progress("Inflating", 0.3)
    dist = ndi.distance_transform_edt(mask)
    rd = max(dist.max() * 0.9, 1.0)
    t = np.clip(dist / rd, 0, 1)
    dome = ndi.gaussian_filter(np.sqrt(1 - (1 - t) ** 2), 1.5)
    thick = puff * side * 0.5
    edge = max(2.0, 0.02 * side)
    dark = ndi.gaussian_filter(1.0 - lum, 0.8)
    engrave = detail * 0.03 * side * np.clip((dark - 0.15) / 0.6, 0, 1)
    h_in = np.maximum(edge + thick * dome - engrave * (dome > 0.02), edge * 0.5)
    h_out = edge - 3.0 * ndi.distance_transform_edt(~mask)
    h = np.where(mask, h_in, h_out).astype(np.float32)

    return heightfield_to_mesh(h, rgb, flat_back, progress)


def heightfield_to_mesh(h: np.ndarray, rgb: np.ndarray, flat_back: bool = True,
                        progress=lambda *_: None) -> trimesh.Trimesh:
    """Turn a height map (pixels, <=0 outside the subject) into a closed, watertight solid."""
    progress("Building solid", 0.5)
    zr = int(np.ceil(h.max())) + 3
    zs = np.arange(-zr, zr + 1, dtype=np.float32)
    if flat_back:
        zs = zs[zs >= -2]
        vol = np.minimum(h[:, :, None] - zs[None, None, :], zs[None, None, :])
    else:
        vol = h[:, :, None] - np.abs(zs)[None, None, :]
    verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=0.0)

    progress("Colouring", 0.8)
    yi = np.clip(np.round(verts[:, 0]).astype(int), 0, rgb.shape[0] - 1)
    xi = np.clip(np.round(verts[:, 1]).astype(int), 0, rgb.shape[1] - 1)
    cols = np.c_[rgb[yi, xi], np.full(len(verts), 255)].astype(np.uint8)
    out = np.c_[verts[:, 1], -verts[:, 0], verts[:, 2] + zs[0]]
    m = trimesh.Trimesh(vertices=out, faces=faces, vertex_colors=cols, process=True)
    m.update_faces(m.nondegenerate_faces())      # marching cubes can emit a few zero-area triangles
    m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:                            # drop tiny specks
        m = max(parts, key=lambda p: len(p.faces))
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m
