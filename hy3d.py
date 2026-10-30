#!/usr/bin/env python3
"""Client for the remote Hunyuan3D worker running in a Kaggle notebook.

    python3 hy3d.py submit photo.png --wait          # image -> textured outputs/<job>/result.glb
    python3 hy3d.py submit photo.png --no-texture --model v2-turbo   # quick untextured draft
    python3 hy3d.py status                           # worker heartbeat + recent jobs
    python3 hy3d.py wait <job_id>
    python3 hy3d.py fetch <job_id>
    python3 hy3d.py preview outputs/<job>/result.glb # PNG turntable (needs trimesh, matplotlib)
    python3 hy3d.py clean --done                     # drop finished jobs from the queue branch

Talks to the worker only through the ``queue`` branch of this repo's ``origin``
remote (see hy3d_queue.py), so it needs nothing but git push access.
"""

import argparse
import json
import secrets
import shutil
import sys
import time
from pathlib import Path

from hy3d_queue import Queue, git, now

ROOT = Path(__file__).resolve().parent
QUEUE_DIR = ROOT / ".hy3d-queue"
OUT_DIR = ROOT / "outputs"
MODELS = ["v2", "v2-turbo", "mini", "mini-turbo", "v2-1"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def open_queue():
    url = git(ROOT, "remote", "get-url", "origin").stdout.strip()
    return Queue(QUEUE_DIR, url, name="hy3d-client", email="hy3d-client@localhost").ensure()


ENGINE_PREFERENCE = ["trellis2", "hunyuan"]  # best first
ALIVE_SECONDS = 20 * 60  # idle workers heartbeat every 10 min


def seconds_ago(ts):
    try:
        return time.time() - (time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone)
    except (TypeError, ValueError):
        return None


def age(ts):
    s = seconds_ago(ts)
    if s is None:
        return "?"
    s = int(s)
    return f"{s}s" if s < 120 else f"{s // 60}m" if s < 7200 else f"{s // 3600}h"


def heartbeats(q):
    """engine name -> latest heartbeat, one file per engine (plus the legacy single file)."""
    beats = {}
    root = q.dir / "worker"
    for p in sorted(root.glob("heartbeat*.json")) if root.exists() else []:
        hb = q.read_json(f"worker/{p.name}") or {}
        eng = hb.get("engine", "hunyuan")
        if eng not in beats or (seconds_ago(hb.get("time")) or 1e9) < (seconds_ago(beats[eng].get("time")) or 1e9):
            beats[eng] = hb
    return beats


def alive(hb):
    s = seconds_ago(hb.get("time"))
    return s is not None and s < ALIVE_SECONDS


def pick_engine(requested, beats):
    if requested != "auto":
        return requested
    for eng in ENGINE_PREFERENCE:
        if eng in beats and alive(beats[eng]):
            return eng
    return "any"


def cmd_submit(a):
    img = Path(a.image)
    if not img.is_file():
        sys.exit(f"no such image: {img}")
    ext = img.suffix.lower()
    if ext not in IMAGE_EXTS:
        sys.exit(f"unsupported image type {ext}; use one of {sorted(IMAGE_EXTS)}")
    q = open_queue()
    beats = heartbeats(q)
    engine = pick_engine(a.engine, beats)
    job_id = time.strftime("%Y%m%d-%H%M%S-", time.gmtime()) + secrets.token_hex(2)
    job = {
        "id": job_id,
        "created": now(),
        "engine": engine,
        "image": f"input{ext}",
        "source_name": img.name,
        "options": {
            "model": a.model,
            "steps": a.steps,
            "octree_resolution": a.octree,
            "guidance_scale": a.guidance,
            "seed": a.seed,
            "texture": not a.no_texture,
            "paint": a.paint,
            "max_faces": a.max_faces,
            "remove_bg": not a.no_rembg,
            "resolution": a.resolution,
            "texture_size": a.texture_size,
        },
    }
    q.publish(
        {f"jobs/{job_id}/job.json": json.dumps(job, indent=2), f"jobs/{job_id}/input{ext}": img},
        f"submit {job_id}",
    )
    print(f"submitted {job_id} for engine {engine}")
    hb = beats.get(engine) if engine != "any" else None
    if engine == "any":
        print("warning: no worker is alive; start molab/trellis_worker.py (or the Kaggle notebook)")
    elif not hb:
        print(f"warning: no {engine} worker has ever checked in")
    else:
        print(f"{engine} worker last seen {age(hb.get('time'))} ago ({hb.get('state', '?')})")
        if not a.no_texture and hb.get("texture") is False:
            print("warning: worker has no texture support (TEXTURE = False in the notebook); "
                  "this job will fail. Resubmit with --no-texture or rerun the notebook with TEXTURE = True")
    if a.wait:
        a.job = job_id
        return cmd_wait(a)


def download(q, job_id, out):
    src = q.dir / "jobs" / job_id
    dst = Path(out) / job_id
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in src.iterdir():
        if p.name.startswith("result") or p.name in ("status.json", "log.txt"):
            shutil.copyfile(p, dst / p.name)
            copied.append(dst / p.name)
    return copied


def cmd_wait(a):
    q = open_queue()
    deadline = time.time() + a.timeout
    last = None
    while True:
        st = q.status(a.job) or {"state": "pending"}
        key = (st.get("state"), st.get("stage"))
        if key != last:
            print(f"[{time.strftime('%H:%M:%S')}] {a.job}: {st.get('state')}"
                  + (f" / {st['stage']}" if st.get("stage") else ""), flush=True)
            last = key
        if st.get("state") == "done":
            for p in download(q, a.job, a.out):
                print(f"  -> {p}")
            return 0
        if st.get("state") == "error":
            print(st.get("error", "unknown error"))
            download(q, a.job, a.out)
            return 1
        if time.time() > deadline:
            print(f"timed out after {a.timeout}s; run `hy3d.py wait {a.job}` to keep waiting")
            return 2
        time.sleep(a.interval)
        q.sync()


def cmd_fetch(a):
    q = open_queue()
    st = q.status(a.job)
    if not st or st.get("state") != "done":
        sys.exit(f"{a.job} is not done: {st}")
    for p in download(q, a.job, a.out):
        print(p)


def cmd_status(a):
    q = open_queue()
    beats = heartbeats(q)
    for eng, hb in beats.items():
        print(f"worker {eng}: {hb.get('state') if alive(hb) else 'OFFLINE'} (seen {age(hb.get('time'))} ago, "
              f"gpu={hb.get('gpu')}, texture={hb.get('texture', '?')}, code={hb.get('code', '?')})")
    if not beats:
        print("worker: never seen")
    jobs = [a.job] if a.job else q.jobs()[-a.limit:]
    for j in jobs:
        st = q.status(j) or {"state": "pending"}
        opts = (q.job(j) or {}).get("options", {})
        extra = st.get("stage") or st.get("error", "")[:80]
        print(f"{j}  {st.get('state'):8} {st.get('engine') or opts.get('model', ''):10} "
              f"{'tex' if opts.get('texture') else '   '}  {extra}")


def cmd_clean(a):
    q = open_queue()
    targets = a.jobs or []
    if a.done:
        targets += [j for j in q.jobs() if (q.status(j) or {}).get("state") in ("done", "error")]
    if not targets:
        print("nothing to clean")
        return
    q.publish({}, f"clean {len(targets)} jobs", remove=[f"jobs/{j}" for j in targets])
    print(f"removed {len(targets)} jobs")


def cmd_reset_queue(a):
    """Replace the queue branch with a fresh orphan commit (drops history of old GLBs)."""
    if not a.yes:
        sys.exit("this force-pushes the queue branch and deletes all jobs; pass --yes")
    q = open_queue()
    git(q.dir, "checkout", "-q", "--orphan", "fresh")
    git(q.dir, "rm", "-rqf", "--ignore-unmatch", "jobs")
    git(q.dir, "add", "-A")
    git(q.dir, "commit", "-qm", "reset queue")
    git(q.dir, "push", "-qf", "origin", f"fresh:{q.branch}")
    git(q.dir, "checkout", "-q", "-B", q.branch, "fresh")
    git(q.dir, "branch", "-qD", "fresh")
    print("queue reset")


def cmd_preview(a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import trimesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        sys.exit("preview needs: pip install trimesh matplotlib")
    mesh = trimesh.load(a.glb, force="mesh")
    faces = mesh.faces
    try:  # sample the texture (or vertex colors) per face
        vc = mesh.visual.to_color().vertex_colors[:, :3] / 255.0
        fc = vc[faces].mean(axis=1)
        textured = mesh.visual.kind is not None and not np.allclose(fc, fc[0])
    except Exception:
        textured = False
    if not textured:
        fc = np.tile([0.72, 0.77, 0.84], (len(faces), 1))
    if len(faces) > a.faces:
        idx = np.random.default_rng(0).choice(len(faces), a.faces, replace=False)
        faces, fc = faces[idx], fc[idx]
    v = mesh.vertices - mesh.bounds.mean(axis=0)
    v = v / (abs(v).max() or 1)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)  # GLB is Y-up; matplotlib is Z-up
    tris = v[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    out = Path(a.out or Path(a.glb).with_suffix(".preview.png"))
    views = [(15, -60), (15, 30), (15, 120), (15, 210)]
    fig = plt.figure(figsize=(4 * len(views), 4.4))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        e, z = np.radians(elev), np.radians(azim)
        eye = np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
        shade = 0.45 + 0.55 * np.abs(n @ eye)  # headlight lambert
        ax.add_collection3d(Poly3DCollection(tris, facecolors=np.clip(fc * shade[:, None], 0, 1),
                                             edgecolor="none", linewidth=0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off()
    fig.suptitle(f"{Path(a.glb).name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces"
                 + (", textured" if textured else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="queue an image for 3D generation")
    s.add_argument("image")
    s.add_argument("--engine", choices=["auto", *ENGINE_PREFERENCE, "any"], default="auto",
                   help="which worker should take the job (auto = best one currently online)")
    s.add_argument("--model", choices=MODELS, default="v2",
                   help="shape model preset (default v2; turbo variants are ~5x faster)")
    s.add_argument("--steps", type=int, help="diffusion steps (default: 30, or 5 for turbo)")
    s.add_argument("--octree", type=int, default=256, help="mesh resolution: 256 fast, 380 detailed")
    s.add_argument("--guidance", type=float, default=5.0)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--no-texture", action="store_true", help="shape only, skip texture painting")
    s.add_argument("--paint", choices=["full", "turbo"], default="full",
                   help="Hunyuan paint model: full (better) or turbo (faster)")
    s.add_argument("--max-faces", type=int,
                   help="decimate to this many faces (0 = keep; default 40k hunyuan, 500k trellis2)")
    s.add_argument("--resolution", type=int, choices=[512, 1024, 1536], default=1024,
                   help="trellis2 only: voxel resolution (1536 = most detail, slowest)")
    s.add_argument("--texture-size", type=int, default=4096, help="trellis2 only: texture map size in px")
    s.add_argument("--no-rembg", action="store_true", help="skip background removal")
    s.add_argument("--wait", action="store_true", help="block until the result is downloaded")

    w = sub.add_parser("wait", help="wait for a job and download its result")
    w.add_argument("job")
    for x in (s, w):
        x.add_argument("--timeout", type=int, default=1800)
        x.add_argument("--interval", type=int, default=15)
        x.add_argument("--out", default=str(OUT_DIR))

    f = sub.add_parser("fetch", help="download a finished job")
    f.add_argument("job")
    f.add_argument("--out", default=str(OUT_DIR))

    st = sub.add_parser("status", help="show worker heartbeat and recent jobs")
    st.add_argument("job", nargs="?")
    st.add_argument("--limit", type=int, default=10)

    c = sub.add_parser("clean", help="remove jobs from the queue branch")
    c.add_argument("jobs", nargs="*")
    c.add_argument("--done", action="store_true", help="remove all done/errored jobs")

    r = sub.add_parser("reset-queue", help="force-reset the queue branch (frees repo space)")
    r.add_argument("--yes", action="store_true")

    pv = sub.add_parser("preview", help="render a GLB to a PNG contact sheet")
    pv.add_argument("glb")
    pv.add_argument("--out")
    pv.add_argument("--faces", type=int, default=150000, help="subsample above this many faces for plotting speed")

    a = p.parse_args()
    handler = {"submit": cmd_submit, "wait": cmd_wait, "fetch": cmd_fetch, "status": cmd_status,
               "clean": cmd_clean, "reset-queue": cmd_reset_queue, "preview": cmd_preview}[a.cmd]
    sys.exit(handler(a) or 0)


if __name__ == "__main__":
    main()
