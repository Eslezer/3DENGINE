# 3DENGINE: handoff notes for Claude

The user sends images in chat (mostly anime characters, often a front and a back view) and wants
**textured 3D models that are usable off the bat**, with no per-model manual cleanup. They are often on a
phone. This repo lets a Claude Code cloud session (which cannot reach GPUs or Hugging Face) drive GPU workers
in notebooks the user runs, through a git queue. Read README.md for setup and option tables; this file is the
"how we work and what we learned" part.

## Architecture (one paragraph)

`hy3d.py` (client, stdlib only) pushes jobs to the orphan **`queue`** branch (`jobs/<id>/job.json`,
`input.<ext>`, optional `back.<ext>`). A worker (`worker.py`) in the user's notebook polls that branch, claims
jobs, pushes `status.json` stages, and finally pushes `result.glb` (+ `result_debug_*.png`), with a heartbeat
per engine (`worker/heartbeat-<engine>.json`). The worker self-updates from `main` when idle (`git pull
--ff-only` and re-exec), so **pushing to main deploys code**: wait until `hy3d.py status` shows the new
`code=<sha>` before submitting a job that needs it. Jobs carry `engine` (`auto`/`trellis2`/`hunyuan`) so two
workers never take the same job. The **`wheels`** branch holds prebuilt CUDA extensions for Blackwell
(`sm120/cp311|cp312|cp313/*.whl` for TRELLIS.2, `hy21/sm120/cp313/` for Hunyuan3D-2.1 paint).

Workers:
- **molab** (marimo, RTX PRO 6000 Blackwell 96 GB): `molab/trellis_worker.py` → `worker.py --engine trellis2`.
  This is the main one. Tokens go in the notebook's password boxes (GITHUB_TOKEN fine-grained with Contents
  read/write on this repo; HF_TOKEN with DINOv3 and FLUX.2-klein licences accepted). It keeps running in the
  background; molab stops it after 12 h or 90 min idle.
- **Kaggle** (2×T4): `kaggle/hy3d_worker.py` → Hunyuan3D-2 shape + paint. Old and lower quality; rarely used now.

Never ask the user to paste tokens in chat.

## Standard workflow for a new character

1. Find the image(s) on disk: `/mnt/user-data/uploads/` or `/root/.claude/uploads/<session>/`. If an image
   is only inline, ask for it as a file.
2. `python3 hy3d.py status`. If the trellis2 heartbeat is older than about 15 min, ask the user to (re)start
   `molab/trellis_worker.py`. Queued jobs run once it starts.
3. Generate (in the background, about 5 min; the first run downloads weights):
   `python3 hy3d.py submit <front> --back <back> --generator pixal3d --resolution 1536 --wait`
   (Pixal3D on the trellis2 worker; with `--back` it reconstructs from both drawings). Alternative:
   `--engine trellis2 --painter hy21 --resolution 1536` (TRELLIS.2 shape + Hunyuan 2.1 paint, about 12 min).
4. Put the drawings back on: `python3 tools/remix.py outputs/<job>/result.glb <front> outputs/<job>/remix
   --back <back> --name <character>`. This upscales the texture and drawings 4x (anime Real-ESRGAN), projects
   them onto the front- and back-facing surfaces and keeps the painter's texture on the sides. It writes a
   GLB and a zipped OBJ.
5. Inspect **close-ups, honestly**: `node tools/render/render.js <glb> <dir>` (run `npm install` in `tools/render`
   once), then read the PNGs. The user notices close-up problems (chopped fronts, front and back swapped,
   blotches), so look for them before they do. A side-by-side sheet of variants made with PIL works well.
6. Deliver with SendUserFile: the GLB, `*_obj.zip` (**the user wants OBJ with its texture**, never an OBJ
   without one) and a comparison or close-up PNG. Say what is still wrong.

### Ranking of what we tried (Flash character, front + back drawings)

0. **Best: Pixal3D (`--generator pixal3d --back`, 1536) + `tools/remix.py`.** Pixal3D lifts image pixels
   into 3D, so its shape follows the drawing closely: the mask, the emblem relief and the suit seams are sculpted, the
   back matches the back drawing, and remix aligns at a silhouette IoU of 0.93. The emblem is clean
   (no stretching), and the eyes and face are crisp. Its own texture is weak (blank purple eyes, smeared mask), so always remix.
   Flaws: some mask red bleeds onto the hair beside the face (3/4 view), the boots are darker and shinier on the sides, and
   the drawing's pink thigh shading comes along. MIT licence, so commercial use is OK (remix's Real-ESRGAN is BSD).
1. **Next best:** the user's other tool's model (Hunyuan-2.1-like: clean, de-lit, 150k faces) + `tools/remix.py`
   with the user's drawings. If the user sends another tool's OBJ + texture, remix it directly. Crisp anime eyes,
   emblem and suit lines; clean sides. Remaining flaws: the drawing's own shading (pink thigh patches) comes along,
   red flecks at the face outline, and the emblem stretches on the deep chest curve.
2. **Our own full pipeline, hy21 + projection:** TRELLIS.2 1536 shape → `--painter hy21` → remix /
   `project_views --keep-base`. On its own hy21 gives blank white eyes, a mangled emblem, washed-out yellows and
   sometimes a wrong back; the projection fixes those.
3. **`--painter ultratex`** (FLUX.2-klein + UltraTex via headless ComfyUI, with front and back pinned):
   faithful front, but baked-in shading, hip stripes and dark boot bands on the sides.
4. **`tools/project_views.py`** alone (harmonic fill on the sides): sides go smeary.
5. **TRELLIS.2's own texture:** usable at `--resolution 1024`; **at 1536 it is corrupted** (dark and
   blotchy). The client warns about this.
6. Hunyuan3D-2 on Kaggle: softest.

## Licences (tell the user when relevant)

- Commercial use is fine with TRELLIS.2 (MIT), its own texture, and the projection tools.
- UltraTex/FLUX.2-klein is FLUX Non-Commercial. Hunyuan3D-2.1 is under the Tencent Hunyuan Non-Commercial
  licence (not valid in the EU, UK or South Korea).
- Pixal3D is MIT too (code and weights).
- If the user says a model is commercial, use Pixal3D or TRELLIS.2 (1024) with projection/remix, and no
  ultratex or hy21 painter.
- Do not help get around a licence.

## Hard-won lessons (do not relearn these)

- **Pixal3D** (`pixal_runner.py`, subprocess in the TRELLIS.2 env): NAF's feature upsampler needs natten.
  natten is installed **without** its CUDA library (the GPU is hidden during `pip install`; a full build takes
  hours), and `na2d_torch` replaces it: exact neighbourhood attention in plain PyTorch, verified against
  natten. Do not use natten's flex backend: it needs equal QK/V head dims, and its eager mode builds a 512 GiB
  dense mask. Front + back go to `inference_mv` on the 0° and 180° cameras of the example rig. Both drawings
  are scaled so the object fills 1/1.1 of a 1024 frame, as in its single-view preprocessing. Its output also
  needs `face_front` (`turned_around` was true on Flash). Job errors carry the last 80 lines of subprocess
  output (`sh_tail`).

- **Orientation:** TRELLIS.2 sometimes builds the figure facing −Z. `trellis_engine.face_front` compares the
  mesh's geometric creases with the drawing's line art from +Z and −Z (`tools/orient_check.py`) and turns it
  around. Colour-based checks failed. The status shows `turned_around`. Every tool assumes +Y up, front = +Z.
- **Transparency:** `hy3d_queue.has_transparency` counts an image as cut out only if at least 1% of its pixels
  have alpha < 128. Many PNGs have a near-opaque alpha channel, and trusting it skipped background removal,
  which made the front pin shrink.
- **Alignment:** projections fit the drawing to the mesh silhouette (IoU coordinate descent). IoU ≥ 0.85 is OK,
  and about 0.9 is typical for good drawings. A "chopped" front came from drawing/mesh mismatch, not from
  the code.
- **hy21 paint** runs in its own uv venv (Python 3.13) because it pins diffusers 0.30 and transformers 4.46. Our
  runner stubs `bpy`, uses spandrel instead of realesrgan/basicsr, and decimates to 150k with fast_simplification
  instead of its 40k remesh (pymeshlab filters need libOpenGL, which is missing). It also patches upstream so
  a list of reference images (front, back) works. Its GLB has no NORMAL attribute; project_views computes
  normals.
- **TRELLIS.2 on Blackwell:** torch 2.11 cu130 from PyPI; xformers replaced by an SDPA shim
  (`trellis_compat/`); BiRefNet instead of gated RMBG-2.0; DINOv3 is gated (HF_TOKEN).
  `tools/build_trellis_wheels.sh` rebuilds the wheels with pip CUDA 13.0 (pin 13.0, add cudart/lib64
  symlinks and a libcuda stub).
- **UltraTex:** ComfyUI runs as a subprocess. `_kill_stale` frees the port from old servers. Rendering falls
  back to nvdiffrast. The `Hy3dPinFront` node pins the front/back views (RePaint keep mask with
  facing falloff).
- Don't `pkill -f` patterns that match your own shell. When git pushes to the queue branch race, retry.

## Tools

| file | what |
|---|---|
| `hy3d.py` | client: `submit / wait / fetch / status / clean / reset-queue / preview / project` |
| `worker.py`, `hy3d_queue.py` | worker loop, git queue |
| `trellis_engine.py` | TRELLIS.2 setup and inference, orientation fix, generators (`trellis2`, `pixal3d`), painters (`trellis`, `ultratex`, `hy21`) |
| `pixal_runner.py` | Pixal3D single- and multi-view runner (patches: SDPA attention, `na2d_torch`, open BiRefNet) |
| `hy21_engine.py`, `hy21_runner.py` | Hunyuan3D-2.1 PBR painter (separate venv, subprocess) |
| `ultratex_engine.py`, `ultratex_compat/` | UltraTex via ComfyUI, nvdiffrast renderer, pin node |
| `tools/remix.py` | **upscale and project the drawings onto any textured GLB/OBJ, keeping its own texture on the sides; GLB + OBJ zip** |
| `tools/upscale.py` | 4x Real-ESRGAN (anime) via spandrel. Needs `torch` (CPU wheel is fine) + `spandrel` |
| `tools/project_views.py` | front/back projection; `--keep-base` keeps the model's texture on the sides, otherwise a harmonic fill |
| `tools/project_front.py` | front-only projection (`hy3d.py project`), GLB helpers, silhouette alignment |
| `tools/orient_check.py` | front/back detection from creases |
| `tools/render/` | three.js close-up renderer (`views.json`; `CLAY=1` for untextured) |

Local dependencies for the tools: `pip install -q --break-system-packages trimesh scipy pygltflib pillow
matplotlib spandrel` and `pip install -q --break-system-packages torch --index-url
https://download.pytorch.org/whl/cpu`, then `npm install` in `tools/render`. If system pip is awkward, a
`uv venv` works too.

## Ideas not done yet

- Remove the drawing's shading before projecting (flatten the soft gradients to the base colours), which
  would fix the pink thigh patches.
- Locally warp the drawing to the mesh's normal-map edges so details like the emblem stay where they belong
  on deep curves.
- Add a 45° view to the projection, or run a second UltraTex pass at 45°.
- Run remix on the worker (GPU) instead of locally, as a `--remix` job option.
- Stop idle workers automatically.

## Housekeeping

`outputs/` and `.hy3d-queue/` are gitignored. Never commit them to `main`. Run
`python3 hy3d.py clean --done` now and then, and `reset-queue --yes` when the repo grows. On a job
`error`, read `outputs/<job>/log.txt`. For CUDA OOM, use a lower `--max-faces` or `--resolution`.
