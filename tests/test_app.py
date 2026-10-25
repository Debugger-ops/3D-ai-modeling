import io
import os
import time

os.environ["I3D_ENGINE"] = "demo"

import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.server import app

client = TestClient(app)


def _png():
    b = io.BytesIO()
    Image.new("RGB", (200, 150), (200, 80, 40)).save(b, "PNG")
    return b.getvalue()


def _run_job():
    r = client.post("/api/jobs", files={"file": ("t.png", _png(), "image/png")}, data={"remove_bg": "false"})
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    return jid, j


def test_health():
    assert client.get("/api/health").json()["ready"] is True


def test_job_and_exports():
    jid, j = _run_job()
    assert j["stats"]["faces"] > 1000
    for fmt in ("glb", "obj", "ply", "stl"):
        r = client.get(f"/api/jobs/{jid}/export", params={"format": fmt, "height_mm": 80})
        assert r.status_code == 200 and len(r.content) > 1000, fmt
    r = client.get(f"/api/jobs/{jid}/export", params={"format": "stl", "height_mm": 80})
    assert r.headers["x-watertight"] == "true"
    m = trimesh.load(io.BytesIO(r.content), file_type="stl")
    assert abs(m.extents[2] - 80) < 0.01          # Z-up, 80 mm tall
    assert abs(m.bounds[0][2]) < 1e-6             # sits on the bed


def test_bad_input():
    assert client.post("/api/jobs", files={"file": ("x.png", b"nope", "image/png")}).status_code == 400
    assert client.get("/api/jobs/missing").status_code == 404
    jid, _ = _run_job()
    assert client.get(f"/api/jobs/{jid}/export", params={"format": "xyz"}).status_code == 400


def _drawing():
    from PIL import ImageDraw
    b = io.BytesIO()
    im = Image.new("RGB", (240, 200), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((30, 30, 210, 170), outline="black", width=4)
    for x in range(60, 200, 20):
        d.line((x, 60, x + 10, 140), fill="black", width=2)
    im.save(b, "PNG")
    return b.getvalue()


def test_inflate_job_is_printable_flat():
    r = client.post("/api/jobs", files={"file": ("d.png", _drawing(), "image/png")},
                    data={"method": "inflate", "resolution": "128", "flat_back": "true"})
    jid = r.json()["id"]
    for _ in range(300):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    assert j["stats"]["watertight"] is True
    r = client.get(f"/api/jobs/{jid}/export", params={"format": "stl", "height_mm": 80, "lay_flat": "true"})
    assert r.headers["x-watertight"] == "true"
    m = trimesh.load(io.BytesIO(r.content), file_type="stl")
    assert abs(max(m.extents[0], m.extents[1]) - 80) < 0.05
    assert m.extents[2] < 40 and abs(m.bounds[0][2]) < 1e-6   # lies flat on the bed


def test_leaky_mesh_gets_repaired():
    from app import postprocess as pp
    s = trimesh.creation.icosphere(subdivisions=3)
    leaky = trimesh.Trimesh(s.vertices, s.faces[40:], process=False)   # remove a patch of faces
    assert not leaky.is_watertight
    data, _, _, st = pp.export_bytes(leaky, "stl", 50)
    assert st["watertight"] is True and st["repair"] in ("holes-filled", "voxel-remesh")


def test_cloud_engine_with_fake_gradio(tmp_path, monkeypatch):
    """Simulates the hosted TRELLIS Space: checks our call sequence and GLB -> coloured mesh loading."""
    import sys
    import types

    import numpy as np
    from PIL import Image as PILImage

    tex = PILImage.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8"))
    box = trimesh.creation.icosphere(subdivisions=2)
    uv = (box.vertices[:, :2] / 2 + 0.5)
    box.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
    glb = tmp_path / "out.glb"
    box.export(str(glb))
    png = tmp_path / "pre.png"
    PILImage.new("RGB", (8, 8), "white").save(png)
    calls = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, api_name=None, **kw):
            calls.append(api_name)
            if api_name == "/preprocess_image":
                return str(png)
            if api_name == "/extract_glb":
                return ({"path": str(glb)}, str(glb))
            return None

    fake = types.ModuleType("gradio_client")
    fake.Client, fake.handle_file = FakeClient, (lambda p: p)
    monkeypatch.setitem(sys.modules, "gradio_client", fake)
    from app import cloud
    m = cloud.generate(PILImage.new("RGBA", (64, 64), (200, 50, 50, 255)))
    assert calls == ["/start_session", "/preprocess_image", "/image_to_3d", "/extract_glb"]
    assert m.is_watertight and len(m.faces) > 100 and m.visual.vertex_colors.shape[1] == 4


def test_edits_info_and_import():
    jid, j = _run_job()
    base = client.get(f"/api/jobs/{jid}/info", params={"height_mm": 60}).json()
    assert base["watertight"] and base["volume_cm3"] > 0 and len(base["view_size"]) == 3
    dec = client.get(f"/api/jobs/{jid}/info", params={"height_mm": 60, "decimate": 30}).json()
    assert dec["faces"] < base["faces"] * 0.5
    pl = client.get(f"/api/jobs/{jid}/info", params={"height_mm": 60, "plinth": 5}).json()
    assert pl["size"][2] > base["size"][2] + 4
    cut = client.get(f"/api/jobs/{jid}/info", params={"height_mm": 60, "cut": 20}).json()
    assert cut["watertight"]
    mir = client.get(f"/api/jobs/{jid}/export", params={"format": "glb", "mirror": "true", "rx": 90, "ry": 90})
    assert mir.status_code == 200 and mir.headers["x-watertight"] == "true"
    # import the exported GLB as a new model
    r = client.post("/api/import", files={"file": ("m.glb", mir.content, "model/gltf-binary")})
    assert r.status_code == 200 and r.json()["status"] == "done"
    assert client.get(f"/api/jobs/{r.json()['id']}/info").json()["faces"] > 1000
    assert client.post("/api/import", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400


def test_depth_sculpt_watertight_with_fake_depth():
    import numpy as np
    from PIL import Image as PILImage
    from app import depth

    def fake_depth(img, progress=None):             # a ramp + bumps, like a real depth map
        h, w = img.size[1], img.size[0]
        yy, xx = np.mgrid[0:h, 0:w]
        return (0.5 * xx / w + 0.3 * (np.sin(xx / 6.0) * np.sin(yy / 6.0) > 0)).astype("float32")

    im = PILImage.open(io.BytesIO(_drawing()))
    m = depth.generate(im, 160, 0.3, 0.8, True, depth_fn=fake_depth)
    assert m.is_watertight and len(m.faces) > 5000
    flat = depth.generate(im, 160, 0.3, 0.0, True, depth_fn=fake_depth)
    assert m.extents[2] != flat.extents[2] or abs(m.volume - flat.volume) > 1
