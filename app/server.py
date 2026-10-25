"""Image to 3D Studio - local web server.   Run:  ./run.sh   (or: uvicorn app.server:app)"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import cloud, depth, inflate
from . import postprocess as pp

try:  # iPhone/Mac photos (HEIC/HEIF) if pillow-heif is installed
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass
from .engine import get_engine

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 25 * 1024 * 1024

app = FastAPI(title="Image to 3D Studio")
jobs: dict[str, dict] = {}
q: "queue.Queue[str]" = queue.Queue()


def _set(job, **kw):
    job.update(kw)


def worker():
    while True:
        jid = q.get()
        job = jobs[jid]
        d = OUT / jid
        t0 = time.time()
        _set(job, status="running", stage="Starting", progress=0.02, started=t0)

        def progress(stage, frac):
            _set(job, stage=stage, progress=frac)

        try:
            img = Image.open(d / "input.png")
            p = job["params"]
            if p["method"] == "inflate":
                px = {128: 240, 256: 320, 320: 420}.get(p["resolution"], 320)
                mesh = inflate.generate(img, px, p["puff"], p["detail"], p["flat_back"], progress)
            elif p["method"] == "depth":
                px = {128: 240, 256: 360, 320: 460}.get(p["resolution"], 360)
                mesh = depth.generate(img, px, p["puff"], p["detail"], p["flat_back"], progress)
            elif p["method"] == "cloud":
                mesh = cloud.generate(img, progress)
            else:
                mesh = get_engine().generate(img, p["remove_bg"], p["fg_ratio"], p["resolution"], progress)
            progress("Saving", 0.95)
            mesh.export(str(d / "raw.ply"))
            _set(job, status="done", stage="Done", progress=1.0, finished=time.time(), stats=pp.stats(mesh))
        except Exception as e:  # surface any failure to the UI
            traceback.print_exc()
            _set(job, status="error", stage="Failed", error=str(e), finished=time.time())
        finally:
            (d / "job.json").write_text(json.dumps({k: v for k, v in job.items()}, default=str))


threading.Thread(target=worker, daemon=True).start()


def _job(jid: str) -> dict:
    if jid not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[jid]


@app.get("/api/health")
def health():
    return get_engine().status()


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), remove_bg: bool = Form(True),
                     fg_ratio: float = Form(0.85), resolution: int = Form(256),
                     method: str = Form("ai"), puff: float = Form(0.30), detail: float = Form(0.5),
                     flat_back: bool = Form(True)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 25 MB)")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"That file is not a readable image ({type(e).__name__}: {e}). Try a JPG or PNG.")
    img.thumbnail((1024, 1024))
    jid = uuid.uuid4().hex[:10]
    d = OUT / jid
    d.mkdir(parents=True)
    img.save(d / "input.png")
    jobs[jid] = {
        "id": jid, "name": file.filename, "status": "queued", "stage": "Queued", "progress": 0.0,
        "created": time.time(), "error": None, "stats": None,
        "params": {"remove_bg": remove_bg, "fg_ratio": min(1.0, max(0.5, fg_ratio)),
                   "resolution": min(384, max(64, resolution)),
                   "method": method if method in ("inflate", "depth", "cloud") else "ai",
                   "puff": min(0.6, max(0.05, puff)), "detail": min(1.0, max(0.0, detail)),
                   "flat_back": flat_back},
    }
    q.put(jid)
    return jobs[jid]


@app.get("/api/jobs")
def list_jobs():
    return sorted(jobs.values(), key=lambda j: j["created"], reverse=True)


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _job(jid)


@app.get("/api/jobs/{jid}/input.png")
def input_png(jid: str):
    _job(jid)
    return FileResponse(OUT / jid / "input.png", media_type="image/png")


def edits(clean: bool = True, smooth: int = 0, rx: int = 0, ry: int = 0, rz: int = 0, mirror: bool = False,
          decimate: int = 100, cut: float = 0.0) -> dict:
    """Non-destructive edit settings shared by preview, info and export."""
    return {"clean": clean, "smooth": min(30, max(0, smooth)), "rx": rx % 360, "ry": ry % 360, "rz": rz % 360,
            "mirror": mirror, "decimate_pct": min(100, max(5, decimate)), "cut": min(40.0, max(0.0, cut))}


def _mesh(jid, e: dict):
    job = _job(jid)
    if job["status"] != "done":
        raise HTTPException(409, "Model not ready yet")
    return pp.build(pp.load_raw(OUT / jid / "raw.ply"), **e)


@app.get("/api/jobs/{jid}/export")
def export(jid: str, format: str = "glb", height_mm: float = 100.0, lay_flat: bool = False,
           plinth: float = 0.0, e: dict = Depends(edits)):
    m = _mesh(jid, e)
    try:
        data, mime, ext, st = pp.export_bytes(m, format, min(2000.0, max(1.0, height_mm)), lay_flat,
                                              min(50.0, max(0.0, plinth)))
    except ValueError as err:
        raise HTTPException(400, str(err))
    return Response(data, media_type=mime, headers={
        "Content-Disposition": f'attachment; filename="model-{jid}.{ext}"',
        "X-Watertight": str(st["watertight"]).lower(),
        "X-Size-Mm": "x".join(map(str, st["size"])),
        "X-Repair": str(st.get("repair")),
        "Access-Control-Expose-Headers": "X-Watertight, X-Size-Mm, X-Repair",
    })


@app.get("/api/jobs/{jid}/info")
def info(jid: str, height_mm: float = 100.0, lay_flat: bool = False, plinth: float = 0.0,
         e: dict = Depends(edits)):
    """Measurements of the model exactly as the STL export would produce it."""
    m = _mesh(jid, e)
    _, _, _, st = pp.export_bytes(m, "stl", min(2000.0, max(1.0, height_mm)), lay_flat, min(50.0, max(0.0, plinth)))
    # size in the viewer's Y-up frame (width, height, depth)
    v = pp.place(m, min(2000.0, max(1.0, height_mm)), z_up=False)
    st["view_size"] = [round(float(x), 2) for x in v.extents]
    st["source"] = _job(jid).get("params", {}).get("method", "import")
    return st


IMPORT_EXT = (".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".3mf")


@app.post("/api/import")
async def import_model(file: UploadFile = File(...)):
    """Bring an existing 3D model (from Meshy, Tripo, Blender, ...) into the workspace."""
    name = file.filename or "model"
    ext = Path(name).suffix.lower()
    if ext not in IMPORT_EXT:
        raise HTTPException(400, f"Unsupported model type '{ext}'. Use one of: {', '.join(IMPORT_EXT)}")
    raw = await file.read()
    if len(raw) > 200 * 1024 * 1024:
        raise HTTPException(413, "Model too large (max 200 MB)")
    try:
        mesh = cloud.glb_to_mesh_from_bytes(raw, ext.lstrip("."))
    except Exception as err:
        raise HTTPException(400, f"Could not read that model ({type(err).__name__}: {err})")
    jid = uuid.uuid4().hex[:10]
    d = OUT / jid
    d.mkdir(parents=True)
    mesh.export(str(d / "raw.ply"))
    Image.new("RGB", (64, 64), (60, 66, 80)).save(d / "input.png")
    jobs[jid] = {"id": jid, "name": name, "status": "done", "stage": "Imported", "progress": 1.0,
                 "created": time.time(), "error": None, "stats": pp.stats(mesh),
                 "params": {"method": "import", "flat_back": False}}
    return jobs[jid]


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
