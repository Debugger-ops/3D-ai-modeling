"""Cloud HQ engine: sends the image to a hosted, state-of-the-art image-to-3D model.

Default: the public TRELLIS demo Space on Hugging Face (free, no GPU needed on your Mac).
  I3D_CLOUD_SPACE   Space id (default "microsoft/TRELLIS")
  HF_TOKEN          optional Hugging Face token - gives you more free GPU quota and a shorter queue

Privacy: the image is uploaded to that third-party service.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import trimesh
from PIL import Image

DEFAULT_SPACES = ["microsoft/TRELLIS", "JeffreyXiang/TRELLIS", "trellis-community/TRELLIS"]
DEFAULT_SPACE = DEFAULT_SPACES[0]


class CloudError(RuntimeError):
    pass


def _first_path(result):
    """Gradio returns str paths, dicts with 'path'/'value', or tuples/lists of those."""
    if isinstance(result, (list, tuple)):
        for item in reversed(result):
            p = _first_path(item)
            if p:
                return p
        return None
    if isinstance(result, dict):
        for k in ("path", "value", "name", "url"):
            if result.get(k):
                return _first_path(result[k])
        return None
    if isinstance(result, str) and result.lower().endswith((".glb", ".gltf", ".obj", ".ply", ".png", ".jpg", ".webp")):
        return result
    return None


def glb_to_mesh_from_bytes(data: bytes, file_type: str) -> trimesh.Trimesh:
    """Load an uploaded model of any common type into one vertex-coloured mesh."""
    import io
    return _flatten(trimesh.load(io.BytesIO(data), file_type=file_type, force="scene"))


def glb_to_mesh(path: str) -> trimesh.Trimesh:
    """Load a (textured) model and flatten it into one vertex-coloured mesh."""
    return _flatten(trimesh.load(path, force="scene"))


def _flatten(scene) -> trimesh.Trimesh:
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)] if hasattr(scene, "dump") else [scene]
    if not meshes:
        raise CloudError("The cloud model returned an empty file")
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    try:
        mesh.visual = mesh.visual.to_color()      # bake texture into per-vertex colours
    except Exception:
        pass
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                           vertex_colors=getattr(mesh.visual, "vertex_colors", None), process=True)


def generate(image: Image.Image, progress=lambda *_: None, seed: int = 0) -> trimesh.Trimesh:
    try:
        from gradio_client import Client, handle_file
    except ImportError as e:
        raise CloudError("gradio_client is not installed - run: pip install gradio_client") from e

    env = os.environ.get("I3D_CLOUD_SPACE", "")
    spaces = [x.strip() for x in env.split(",") if x.strip()] or DEFAULT_SPACES
    token = os.environ.get("HF_TOKEN") or None
    tmp = Path(tempfile.mkdtemp(prefix="i3d_cloud_"))
    src = tmp / "input.png"
    flat = Image.new("RGB", image.size, "white")           # TRELLIS expects an opaque photo
    flat.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[3])
    flat.save(src)

    client, space, errors = None, None, []
    for sp in spaces:
        progress(f"Connecting to {sp}", 0.08)
        try:
            client = Client(sp, hf_token=token, verbose=False) if token else Client(sp, verbose=False)
            space = sp
            break
        except Exception as e:
            errors.append(f"{sp}: {str(e)[:120]}")
    if client is None:
        raise CloudError("No cloud model Space is reachable right now:\n  " + "\n  ".join(errors) +
                         "\nFree Spaces sometimes break or sleep. Try again later, set I3D_CLOUD_SPACE=owner/space "
                         "(comma-separate several), or use the Inflate method meanwhile.")
    try:
        try:
            client.predict(api_name="/start_session")
        except Exception:
            pass                                              # not all versions have it

        progress("Preparing image (cloud)", 0.2)
        processed = _first_path(client.predict(image=handle_file(str(src)), api_name="/preprocess_image"))
        if not processed:
            raise CloudError("preprocess step returned no image")

        progress("Generating 3D shape (cloud queue - can take 1-3 min)", 0.35)
        client.predict(image=handle_file(processed), multiimages=[], seed=seed,
                       ss_guidance_strength=7.5, ss_sampling_steps=12,
                       slat_guidance_strength=3.0, slat_sampling_steps=12,
                       multiimage_algo="stochastic", api_name="/image_to_3d")

        progress("Extracting textured mesh (cloud)", 0.75)
        out = client.predict(mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")
        glb = _first_path(out)
        if not glb:
            raise CloudError(f"extract step returned no model file: {out!r}")
    except CloudError:
        raise
    except Exception as e:
        msg = str(e)
        hint = ""
        if "quota" in msg.lower() or "gpu" in msg.lower():
            hint = " Free GPU quota used up: set HF_TOKEN=<your Hugging Face token> before ./run.sh, or wait and retry."
        elif isinstance(e, (TypeError, ValueError)):
            try:
                hint = "\nAPI of this Space:\n" + str(client.view_api(return_format="str"))[:1500]
            except Exception:
                pass
        raise CloudError(f"Cloud generation failed: {msg[:300]}.{hint}") from e

    progress("Downloading model", 0.9)
    return glb_to_mesh(glb)
