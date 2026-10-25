"""Depth sculpt engine: a monocular depth-estimation AI (runs locally, free, offline after the first download)
reads the drawing/photo and gives every pixel a relative height, so snouts, horns, scales and ridges stand
out from the surface instead of the whole outline just being puffed into one dome.

  I3D_DEPTH_MODEL   Hugging Face model id (default "Intel/dpt-hybrid-midas", ~490 MB, downloaded once)
"""
from __future__ import annotations

import os

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure

from . import inflate

DEFAULT_MODEL = "Intel/dpt-hybrid-midas"
_pipe = None


class DepthError(RuntimeError):
    pass


def estimate(img: Image.Image, progress=lambda *_: None) -> np.ndarray:
    """Relative depth, 0..1, larger = nearer to the camera, same size as img."""
    global _pipe
    try:
        if _pipe is None:
            progress("Loading depth AI (first run downloads ~490 MB)", 0.15)
            from transformers import pipeline
            _pipe = pipeline("depth-estimation", model=os.environ.get("I3D_DEPTH_MODEL", DEFAULT_MODEL))
        progress("Estimating depth", 0.3)
        out = _pipe(img.convert("RGB"))["predicted_depth"]
        d = np.asarray(out.squeeze().cpu() if hasattr(out, "cpu") else out, dtype=np.float32)
    except Exception as e:
        raise DepthError(f"Depth AI failed to load or run: {type(e).__name__}: {str(e)[:200]}. "
                         f"Check your internet for the first download, or use Inflate silhouette.") from e
    d = np.asarray(Image.fromarray(d).resize(img.size, Image.BICUBIC), dtype=np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)


def generate(image: Image.Image, resolution: int = 320, puff: float = 0.30, detail: float = 0.5,
             flat_back: bool = True, progress=lambda *_: None, depth_fn=estimate) -> trimesh.Trimesh:
    """puff: overall thickness. detail: 0..1 how strongly the AI depth (and pen lines) shape the surface."""
    progress("Cutting out silhouette", 0.1)
    img = image.convert("RGBA")
    scale = resolution / max(img.size)
    img = img.resize((max(8, round(img.width * scale)), max(8, round(img.height * scale))), Image.LANCZOS)
    mask = inflate.foreground_mask(img)
    flat = Image.new("RGB", img.size, "white")
    flat.paste(img, mask=img.split()[3])
    d = depth_fn(flat, progress)

    pad = 8
    mask = np.pad(mask, pad)
    d = np.pad(d, pad, mode="edge")
    rgb = np.pad(np.asarray(img.convert("RGB")), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    lum = rgb.astype(np.float32).mean(-1) / 255.0
    side = max(mask.shape)

    progress("Sculpting", 0.55)
    dist = ndi.distance_transform_edt(mask)
    t = np.clip(dist / max(dist.max() * 0.9, 1.0), 0, 1)
    dome = ndi.gaussian_filter(np.sqrt(1 - (1 - t) ** 2), 1.5)
    inside = d[mask]
    dn = np.zeros_like(d)                           # depth normalised inside the subject only
    if inside.size:
        dn = np.clip((d - np.percentile(inside, 2)) / max(np.percentile(inside, 98) - np.percentile(inside, 2), 1e-6), 0, 1)
    dn = exposure.equalize_adapthist(dn, clip_limit=0.02)          # local contrast -> crisper features
    dn = dn + 0.6 * (dn - ndi.gaussian_filter(dn, 3))               # unsharp: sharpen ridges / scales
    dn = np.clip(ndi.gaussian_filter(dn, 0.7), 0, 1)
    dark = ndi.gaussian_filter(1.0 - lum, 0.8)
    lines = np.clip((dark - 0.15) / 0.6, 0, 1)

    thick = puff * side * 0.5
    edge = max(2.0, 0.02 * side)
    relief = detail * thick * 0.9 * dn * dome**0.5                  # AI depth carves the surface...
    engrave = detail * 0.02 * side * lines                           # ...and pen lines stay readable
    h_in = np.maximum(edge + thick * (0.45 + 0.55 * dome) * 0.6 + relief - engrave, edge * 0.5)
    h_out = edge - 3.0 * ndi.distance_transform_edt(~mask)
    h = np.where(mask, h_in, h_out).astype(np.float32)
    return inflate.heightfield_to_mesh(h, rgb, flat_back, progress)
