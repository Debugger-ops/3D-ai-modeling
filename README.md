# Image to 3D Studio

Turn a single photo into a full 360° 3D model (front, back and sides), preview it in the browser, tidy it up, and export STL / OBJ / GLB / PLY.

- **AI engine:** [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (single-image reconstruction), run locally.
- **Backend:** FastAPI (`app/server.py`), job queue, mesh clean-up with trimesh (`app/postprocess.py`).
- **Frontend:** `app/static/index.html` (three.js viewer, no build step). `relief.html` is the quick offline relief mode.

## Setup (macOS, once)

```bash
brew install cmake python@3.11       # skip what you already have
cd ~/Projects/image-to-3d-studio
PYTHON=python3.11 ./install.sh
./run.sh                             # opens http://127.0.0.1:8000
```

First generation downloads ~1.7 GB of weights plus a ~170 MB background-removal model; both are cached after that.

## Two methods

| Method | Best for | Result |
|---|---|---|
| **Inflate silhouette** (default, no AI) | line drawings, logos, sketches, cut-out art | cuts out the outline, puffs it into a solid dome, engraves the dark lines; always watertight; ready to print |
| **Cloud HQ (TRELLIS)** | photos, renders, shaded art - best quality | newer hosted model; textured, much cleaner shapes. Needs internet, image is uploaded to a Hugging Face Space, queue 1-3 min |
| **AI full 3D (TripoSR, local)** | colour photos / shaded renders of one object | full 360° guess incl. the hidden back; lower quality, runs offline |

Flat line art gives poor results with the AI method because it has no shading to infer depth from - use *Inflate* for it.

## The workspace

| Area | What it does |
|---|---|
| Top bar | Import any GLB/OBJ/STL/PLY model; shading (Shaded, Clay, Normals, Wire, X-ray); camera presets, Ortho/Perspective; Snapshot PNG |
| Left | Source image, method (Inflate / Cloud HQ / Local AI), generation settings, session history |
| Right · Model | Rotate X/Y/Z, mirror, remove floaters, smooth, simplify, flat-cut base, plinth |
| Right · Look | Colour/clay, roughness, metalness, tint |
| Right · Scene | Lighting presets, background (incl. transparent), grid, shadow, axes, dimension labels, turntable |
| Right · Info | Triangles, size in mm, volume, area, watertight, printer-bed fit checks |
| Right · Export | STL / OBJ / GLB / PLY, size in mm, print-flat option |

All Model edits are non-destructive: the original generated mesh is kept, edits are re-applied on preview and export.

## Using it

1. Drop an image (one clear object works best; 3/4 front view).
2. Leave *Remove background* on unless the image is already a cut-out.
3. **Generate**. Expect roughly tens of seconds on an NVIDIA GPU and a minute or more on a Mac CPU.
4. Fix orientation with *Rotate 90°*, remove floaters, smooth, then export.
5. STL export tries to close any holes automatically (fill, then voxel re-mesh if needed) and reports the result. Tick *print flat on its back* for plaques/reliefs. STL export is scaled to your chosen height in mm, Z-up, resting on the bed, with holes filled.

## Project layout

```
app/server.py        API + job worker           app/engine.py       TripoSR wrapper (+ demo engine)
app/postprocess.py   clean / rotate / scale / export
app/static/          index.html (studio), relief.html (quick mode)
tests/test_app.py    pytest suite (uses the demo engine)
install.sh run.sh    setup and launch
```

## Developing in VS Code

```bash
code ~/Projects/image-to-3d-studio
```

Use `.venv` as the interpreter. Run the tests without the AI model: `pip install pytest httpx && pytest`.
Test the UI without the AI model: `I3D_ENGINE=demo ./run.sh` (produces a placeholder blob coloured from your image).

## Troubleshooting

| Problem | Fix |
|---|---|
| `torchmcubes` fails to build | `brew install cmake`, install Xcode tools (`xcode-select --install`), re-run `./install.sh` |
| `tokenizers` / `transformers` build errors | You are on Python 3.12+; use `PYTHON=python3.11 ./install.sh` |
| Very slow on Mac | Try `I3D_DEVICE=mps ./run.sh`; or lower *Mesh detail* |
| Model is upside down/sideways | Use *Rotate 90°* before exporting |
| Out of memory | Use *Low* mesh detail and a smaller image |

Single-image reconstruction guesses the hidden back side, so expect plausible rather than exact geometry. For measured accuracy use multi-photo photogrammetry.

## Cloud HQ settings

```bash
HF_TOKEN=hf_xxx ./run.sh                       # optional: free Hugging Face token = more GPU quota, shorter queue
I3D_CLOUD_SPACE=owner/space ./run.sh           # use a different TRELLIS-compatible Space
```

Free Spaces can be busy, asleep or change their API. If Cloud HQ errors, the message says why (and prints the Space's API when the call signature changed).
# 3D-ai-modeling
