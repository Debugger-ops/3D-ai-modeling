"""Image -> 3D mesh engines.

TripoSREngine : real single-image 3D reconstruction (VAST-AI / Stability TripoSR).
DemoEngine    : instant placeholder blob, used for testing the app without the AI model
                (enable with  I3D_ENGINE=demo ).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "TripoSR"
Progress = Callable[[str, float], None]


class EngineError(RuntimeError):
    pass


class TripoSREngine:
    name = "TripoSR"

    def __init__(self):
        self._model = None
        self._rembg = None
        self._device = None
        self._lock = threading.Lock()

    # ---- status (cheap, never loads the model) ----
    def status(self) -> dict:
        problems = []
        if not VENDOR.exists():
            problems.append("TripoSR source missing - run ./install.sh")
        for mod in ("torch", "omegaconf", "einops", "transformers", "rembg", "torchmcubes"):
            if importlib.util.find_spec(mod) is None:
                problems.append(f"Python package '{mod}' not installed - run ./install.sh")
        return {"engine": self.name, "ready": not problems, "problems": problems, "device": self._pick_device(),
                "loaded": self._model is not None}

    def _pick_device(self) -> str:
        forced = os.environ.get("I3D_DEVICE")
        if forced:
            return forced
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"  # set I3D_DEVICE=mps to try Apple GPU (faster, less tested)

    def _load(self, progress: Progress):
        if self._model is not None:
            return
        st = self.status()
        if not st["ready"]:
            raise EngineError("; ".join(st["problems"]))
        sys.path.insert(0, str(VENDOR))
        import rembg
        import torch
        from tsr.system import TSR

        self._device = self._pick_device()
        progress("Loading AI model (first run downloads ~1.7 GB)", 0.05)
        model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
        model.renderer.set_chunk_size(8192)
        model.to(self._device)
        self._model = model
        self._rembg = rembg.new_session()
        torch.set_grad_enabled(False)

    # ---- main entry ----
    def generate(self, image: Image.Image, remove_bg: bool, fg_ratio: float, resolution: int,
                 progress: Progress) -> trimesh.Trimesh:
        with self._lock:
            self._load(progress)
            import torch
            from tsr.utils import remove_background, resize_foreground

            progress("Preparing image", 0.15)
            if remove_bg:
                progress("Removing background", 0.2)
                image = remove_background(image.convert("RGBA"), self._rembg)
            image = image.convert("RGBA")
            image = resize_foreground(image, fg_ratio)
            arr = np.asarray(image).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            rgb = Image.fromarray((arr * 255).astype(np.uint8))

            progress("Reconstructing 3D shape (the slow step)", 0.35)
            with torch.no_grad():
                codes = self._model([rgb], device=self._device)
            progress("Extracting mesh", 0.8)
            with torch.no_grad():
                meshes = self._model.extract_mesh(codes, True, resolution=resolution)
            return meshes[0]


class DemoEngine:
    """Placeholder: an egg-shaped blob coloured from the image. Not AI - for UI/pipeline testing."""
    name = "Demo (no AI)"

    def status(self) -> dict:
        return {"engine": self.name, "ready": True, "problems": [], "device": "cpu", "loaded": True}

    def generate(self, image, remove_bg, fg_ratio, resolution, progress) -> trimesh.Trimesh:
        progress("Building demo mesh", 0.5)
        m = trimesh.creation.icosphere(subdivisions=4)
        v = m.vertices
        im = np.asarray(image.convert("RGB").resize((128, 128)))
        u = ((v[:, 0] * 0.5 + 0.5) * 127).astype(int)
        w = ((0.5 - v[:, 1] * 0.5) * 127).astype(int)
        m.visual.vertex_colors = np.c_[im[w.clip(0, 127), u.clip(0, 127)], np.full(len(v), 255)]
        m.apply_scale([1.0, 1.25, 0.8])
        return m


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = DemoEngine() if os.environ.get("I3D_ENGINE", "triposr").lower() == "demo" else TripoSREngine()
    return _engine
