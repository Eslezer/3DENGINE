# Using this repo as a 3D-generation tool

This repo is a git-backed proxy to an image-to-3D worker running in the user's notebook (see README.md): TRELLIS.2 on molab (`engine=trellis2`, preferred) or Hunyuan3D-2 on Kaggle (`engine=hunyuan`). `hy3d.py status` shows which one is online. When the user sends an image and wants a 3D model:

1. **Get the image onto disk.** Chat attachments usually land in `/mnt/user-data/uploads/`. If the image only appears inline and can't be found on disk, ask the user to attach it as a file.
2. **Check the worker:** `python3 hy3d.py status`. If the heartbeat is older than about 15 min or missing, tell the user to (re)start `molab/trellis_worker.py` on molab (or `kaggle/hy3d_worker.py` on Kaggle). Jobs submitted now are processed once it starts.
3. **Submit:** `python3 hy3d.py submit <image> --wait` (run it in the background; textured jobs take a few minutes). Shape + texture is the default and produces one GLB with the texture embedded. On trellis2, `--resolution 1536` gives the most detail. On hunyuan, `--paint turbo` gives a faster texture and `--no-texture` gives a shape-only draft. If `status` shows `texture=False`, the notebook was set up without texturing.
4. **Inspect:** `pip install -q --break-system-packages trimesh matplotlib` once, then `python3 hy3d.py preview outputs/<job>/result.glb` for a quick look. `preview` averages one color per face, so for real texture detail run `npm install` in `tools/render` once and `node tools/render/render.js outputs/<job>/result.glb outputs/<job>/closeups` (views in `tools/render/views.json`). Read the PNGs and say honestly what looks wrong.
5. **Deliver:** send the user `outputs/<job>/result.glb` plus the preview PNG.
6. On `error`, read `outputs/<job>/log.txt`. CUDA OOM → retry with `--paint turbo`, a lower `--max-faces`, `--model mini`, or `--no-texture`.

`outputs/` and `.hy3d-queue/` are gitignored. Never commit them to `main`. Run `python3 hy3d.py clean --done` now and then.
