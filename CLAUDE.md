# Using this repo as a 3D-generation tool

This repo is a git-backed proxy to a Hunyuan3D-2 worker running in the user's Kaggle notebook (see README.md). When the user sends an image and wants a 3D model:

1. **Get the image onto disk.** Chat attachments usually land in `/mnt/user-data/uploads/`. If the image only appears inline and can't be found on disk, ask the user to attach it as a file.
2. **Check the worker:** `python3 hy3d.py status`. If the heartbeat is older than about 15 min or missing, tell the user to (re)start `kaggle/hy3d_worker.ipynb` on Kaggle. Jobs submitted now are processed once it starts.
3. **Submit:** `python3 hy3d.py submit <image> --wait` (run it in the background for long jobs). Use `--model v2-turbo` for quick drafts and `--texture` only if the user wants color and the notebook was set up with `TEXTURE = True`.
4. **Inspect:** `pip install -q --break-system-packages trimesh matplotlib` once, then `python3 hy3d.py preview outputs/<job>/result.glb`, and Read the PNG to check the result.
5. **Deliver:** send the user `outputs/<job>/result.glb` plus the preview PNG.
6. On `error`, read `outputs/<job>/log.txt`. CUDA OOM → retry without `--texture`, with `--model mini`, or with a lower `--octree`.

`outputs/` and `.hy3d-queue/` are gitignored. Never commit them to `main`. Run `python3 hy3d.py clean --done` now and then.
