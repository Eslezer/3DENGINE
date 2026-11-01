# 3DENGINE: remote image-to-3D workers

Turn images into textured 3D models (a single GLB with the texture embedded) using a GPU notebook you control:

| engine | where | model | notes |
|---|---|---|---|
| `trellis2` | [molab](https://molab.marimo.io) (RTX PRO 6000 Blackwell, 96 GB) | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | best quality: shape and PBR texture in one pass, up to 1536³ voxels and 4K textures |
| `hunyuan` | Kaggle (2× T4, 16 GB) | [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | fits free Kaggle; softer texture |

Either one is driven from anywhere that can `git push` to this repo. That includes a Claude Code cloud session, whose network can reach GitHub but not Kaggle or Hugging Face.

```
 client (hy3d.py)                 GitHub: branch "queue"                 Kaggle notebook (worker.py)
 ───────────────                  ─────────────────────                  ───────────────────────────
 submit photo.png  ── push ──▶  jobs/<id>/job.json + input.png  ◀── poll every 10s
                                jobs/<id>/status.json            ◀── push: claimed → running/<stage>
 wait <id>         ◀── fetch ── jobs/<id>/result.glb             ◀── push: done (or error + log.txt)
                                worker/heartbeat.json            ◀── push every 10 min while idle
```

No tunnels, no open ports. The worker only makes outbound connections to GitHub and Hugging Face.

## 1a. Start the TRELLIS.2 worker on molab

1. GitHub fine-grained token as described below, plus a Hugging Face **read** token. TRELLIS.2 reads images with Meta's DINOv3, which is gated: accept its license at [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) first.
2. Open [`molab/trellis_worker.py`](molab/trellis_worker.py) in molab (it is a marimo notebook) and enable the GPU from the specs button in the header.
3. Paste both tokens into the password boxes and press **Start worker**. Setup installs PyTorch 2.11 (CUDA 13) and the TRELLIS.2 extensions prebuilt for Blackwell from this repo's `wheels` branch; on another GPU or Python version it compiles them instead (30+ min).

molab stops notebooks after 12 h, or after 90 min idle. Queued jobs wait for the next start.

## 1b. Start the Hunyuan worker on Kaggle

1. Create a GitHub **fine-grained token**: *Settings → Developer settings → Fine-grained tokens*, repository access limited to this repo, permission **Contents: Read and write**.
2. On Kaggle: *Create → New Notebook → File → Import Notebook*, then upload [`kaggle/hy3d_worker.ipynb`](kaggle/hy3d_worker.ipynb). On a phone, or if the upload fails, paste the contents of [`kaggle/hy3d_worker.py`](kaggle/hy3d_worker.py) into a single cell instead.
3. Notebook settings: **Accelerator: GPU T4 x2**, **Internet: on**. Under *Add-ons → Secrets*, add `GITHUB_TOKEN` and attach it to the notebook.
4. **Run All**. Setup takes about 15 minutes, mostly building the texture extensions (about 5 if you set `TEXTURE = False` for shapes only). The last cell prints `worker ... polling` and serves jobs until stopped.
   - For a headless worker that keeps going after you close the tab, use *Save Version → Save & Run All (Commit)*. It runs for up to 12 h and draws on your 30 h/week GPU quota.

## 2. Submit jobs

```bash
python3 hy3d.py status                         # is the worker alive?
python3 hy3d.py submit cat.png --wait          # shape + texture → outputs/<job>/result.glb
python3 hy3d.py submit cat.png --paint turbo --wait          # faster texture
python3 hy3d.py submit cat.png --no-texture --model v2-turbo --wait   # quick shape-only draft
python3 hy3d.py preview outputs/<job>/result.glb  # PNG contact sheet (pip install trimesh matplotlib)
python3 hy3d.py project <job>                  # paint the source image onto the front (pip install scipy pygltflib)
python3 tools/remix.py outputs/<job>/result.glb front.png outdir --back back.png   # best finish: sharp drawings on top
python3 hy3d.py clean --done                   # remove finished jobs from the queue branch
python3 hy3d.py reset-queue --yes              # squash queue history to reclaim repo space
```

| option | default | notes |
|---|---|---|
| `--model` | `v2` | `v2`, `v2-turbo` (5 steps + FlashVDM, much faster), `mini`, `mini-turbo` (0.6B, lighter), `v2-1` (Hunyuan3D-2.1 shape model, best detail, more VRAM) |
| `--steps` | 30 / 5 turbo | diffusion steps |
| `--octree` | 256 | mesh grid resolution; 380 gives more detail and is slower |
| `--max-faces` | 40k hunyuan / 500k trellis2 | decimation target; 0 keeps the full mesh |
| `--no-texture` | texture on | skip Hunyuan texture painting |
| `--paint` | `full` | Hunyuan paint model: `full` (better) or `turbo` (faster) |
| `--resolution` | 1024 | trellis2: 512, 1024 or 1536. 1536 gives a finer shape but corrupted TRELLIS.2's own texture in testing, so only use it with `--painter ultratex` |
| `--texture-size` | 4096 | trellis2: texture map size |
| `--painter` | `trellis` | trellis2: `ultratex` repaints TRELLIS.2's mesh with [UltraTex](https://github.com/visualbruno/ComfyUI-UltraTex) (FLUX.2-Klein base 9B, six 2048 px views, headless ComfyUI). FLUX Non-Commercial License; accept it at [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) first. ~25 GB of extra downloads on first use. `hy21` repaints it with [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)'s PBR texture model (albedo with lighting removed, plus metallic/roughness), in its own venv; mesh decimated to 150k faces first. Tencent Hunyuan Non-Commercial License (not valid in the EU, UK or South Korea). Best combined with `tools/remix.py` |
| `--generator` | `trellis2` | trellis2 engine: `pixal3d` builds shape + texture with [Pixal3D](https://github.com/TencentARC/Pixal3D) (TRELLIS.2-based, pixel-aligned to the image, MIT). With `--back` it reconstructs from both views (multi-view pipeline, 0°/180° cameras). Resolution 1024 or 1536 |
| `--no-rembg` | off | skip background removal (it is skipped automatically for images that already have transparency) |
| `--seed`, `--guidance` | 1234, 5.0 | |

Rough T4 timings once weights are cached: shape 30 to 90 s, texture another 1 to 3 min. The first job also downloads several GB of weights. While painting, the worker moves the shape model to CPU so the texture model gets the whole GPU, and falls back to CPU offload if it still runs out of memory.

## Notes

- **Single worker.** On startup the worker re-queues any job left `claimed` or `running` by a previous session, so only run one at a time.
- **Repo growth.** Every GLB is committed to the `queue` branch. `clean` removes files but history keeps them, so run `reset-queue --yes` occasionally. `main` is never touched.
- **Size limit.** Results over 95 MB are rejected (GitHub's per-file limit is 100 MB). Lower `--max-faces` or skip texture.
- The token lives only in Kaggle Secrets and in git config under `/tmp`. It is never written to `/kaggle/working`, which Kaggle saves as notebook output.

## Files

- `hy3d.py`: client CLI (stdlib only; `preview` needs trimesh and matplotlib)
- `worker.py`: the worker; `--engine hunyuan|trellis2`, `--setup` installs the engine
- `hy21_engine.py` / `hy21_runner.py`: Hunyuan3D-2.1 painter (separate Python 3.13 venv; prebuilt extensions on the `wheels` branch under `hy21/`)
- `ultratex_engine.py`: UltraTex painter (installs ComfyUI, downloads models, drives its API); `ultratex_compat/render_nvdr.py`: nvdiffrast fallback for UltraTex's OpenGL renderer on headless GPUs
- `pixal_runner.py`: Pixal3D runner (subprocess in the TRELLIS.2 env; pure-PyTorch neighbourhood attention instead of natten's CUDA kernels)
- `trellis_engine.py`: TRELLIS.2 setup and inference; `trellis_compat/`: torch-SDPA stand-in for xformers
- `molab/trellis_worker.py`: marimo notebook that runs the TRELLIS.2 worker
- `tools/build_trellis_wheels.sh`: builds the Blackwell extension wheels (no GPU needed)
- `tools/remix.py`: upscales a textured GLB/OBJ (any source, including other tools) and the drawings 4x, projects the front/back drawings over it, keeps its texture on the sides; writes GLB + OBJ zip
- `tools/upscale.py`: 4x Real-ESRGAN (anime) via spandrel, CPU or GPU
- `tools/project_views.py`: front + back projection (`--keep-base` keeps the model's texture where the drawings don't reach)
- `tools/orient_check.py`: front vs back detection used by the TRELLIS.2 orientation fix
- `tools/project_front.py`: projects the source image onto the front-facing part of a GLB's texture, keeping generated texture on the back
- `tools/render/`: renders close-up PNGs of a GLB with its real textures (three.js in headless Chromium)
- `hy3d_queue.py`: shared git queue
- `kaggle/hy3d_worker.ipynb`: notebook that runs the worker
- `kaggle/hy3d_worker.py`: the same thing as one paste-into-a-cell script
