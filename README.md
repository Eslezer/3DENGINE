# 3DENGINE: remote Hunyuan3D via Kaggle

Turn images into textured 3D models (a single GLB with the texture embedded) with [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) running on a free Kaggle GPU, driven from anywhere that can `git push` to this repo. That includes a Claude Code cloud session, whose network can reach GitHub but not Kaggle or Hugging Face.

```
 client (hy3d.py)                 GitHub: branch "queue"                 Kaggle notebook (worker.py)
 ───────────────                  ─────────────────────                  ───────────────────────────
 submit photo.png  ── push ──▶  jobs/<id>/job.json + input.png  ◀── poll every 10s
                                jobs/<id>/status.json            ◀── push: claimed → running/<stage>
 wait <id>         ◀── fetch ── jobs/<id>/result.glb             ◀── push: done (or error + log.txt)
                                worker/heartbeat.json            ◀── push every 10 min while idle
```

No tunnels, no open ports. The worker only makes outbound connections to GitHub and Hugging Face.

## 1. Start the worker on Kaggle (once per session)

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
python3 hy3d.py clean --done                   # remove finished jobs from the queue branch
python3 hy3d.py reset-queue --yes              # squash queue history to reclaim repo space
```

| option | default | notes |
|---|---|---|
| `--model` | `v2` | `v2`, `v2-turbo` (5 steps + FlashVDM, much faster), `mini`, `mini-turbo` (0.6B, lighter), `v2-1` (Hunyuan3D-2.1 shape model, best detail, more VRAM) |
| `--steps` | 30 / 5 turbo | diffusion steps |
| `--octree` | 256 | mesh grid resolution; 380 gives more detail and is slower |
| `--max-faces` | 40000 | decimation target; 0 keeps the full mesh |
| `--no-texture` | texture on | skip Hunyuan texture painting |
| `--paint` | `full` | Hunyuan paint model: `full` (better) or `turbo` (faster) |
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
- `worker.py`: Kaggle/GPU worker; `--setup` installs Hunyuan3D-2
- `hy3d_queue.py`: shared git queue
- `kaggle/hy3d_worker.ipynb`: notebook that runs the worker
- `kaggle/hy3d_worker.py`: the same thing as one paste-into-a-cell script
