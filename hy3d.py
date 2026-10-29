#!/usr/bin/env python3
"""Client for the remote Hunyuan3D worker running in a Kaggle notebook.

    python3 hy3d.py submit photo.png --wait          # image -> outputs/<job>/result.glb
    python3 hy3d.py submit photo.png --texture --model v2-turbo
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


def age(ts):
    try:
        t = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except (TypeError, ValueError):
        return "?"
    s = int(time.time() - t)
    return f"{s}s" if s < 120 else f"{s // 60}m" if s < 7200 else f"{s // 3600}h"


def cmd_submit(a):
    img = Path(a.image)
    if not img.is_file():
        sys.exit(f"no such image: {img}")
    ext = img.suffix.lower()
    if ext not in IMAGE_EXTS:
        sys.exit(f"unsupported image type {ext}; use one of {sorted(IMAGE_EXTS)}")
    q = open_queue()
    job_id = time.strftime("%Y%m%d-%H%M%S-", time.gmtime()) + secrets.token_hex(2)
    job = {
        "id": job_id,
        "created": now(),
        "image": f"input{ext}",
        "source_name": img.name,
        "options": {
            "model": a.model,
            "steps": a.steps,
            "octree_resolution": a.octree,
            "guidance_scale": a.guidance,
            "seed": a.seed,
            "texture": a.texture,
            "max_faces": a.max_faces,
            "remove_bg": not a.no_rembg,
        },
    }
    q.publish(
        {f"jobs/{job_id}/job.json": json.dumps(job, indent=2), f"jobs/{job_id}/input{ext}": img},
        f"submit {job_id}",
    )
    print(f"submitted {job_id}")
    hb = q.read_json("worker/heartbeat.json")
    if not hb:
        print("warning: no worker has ever checked in; start the Kaggle notebook")
    else:
        print(f"worker last seen {age(hb.get('time'))} ago ({hb.get('state', '?')})")
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
    hb = q.read_json("worker/heartbeat.json")
    if hb:
        print(f"worker: {hb.get('state')} (seen {age(hb.get('time'))} ago, "
              f"gpu={hb.get('gpu')}, host={hb.get('host')})")
    else:
        print("worker: never seen")
    jobs = [a.job] if a.job else q.jobs()[-a.limit:]
    for j in jobs:
        st = q.status(j) or {"state": "pending"}
        opts = (q.job(j) or {}).get("options", {})
        extra = st.get("stage") or st.get("error", "")[:80]
        print(f"{j}  {st.get('state'):8} {opts.get('model', ''):10} "
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
        import trimesh
    except ImportError:
        sys.exit("preview needs: pip install trimesh matplotlib")
    mesh = trimesh.load(a.glb, force="mesh")
    if len(mesh.faces) > a.faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=a.faces)
        except Exception:
            idx = sorted(__import__("random").Random(0).sample(range(len(mesh.faces)), a.faces))
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[idx], process=False)
    v = mesh.vertices - mesh.bounds.mean(axis=0)
    v = v / (abs(v).max() or 1)
    out = Path(a.out or Path(a.glb).with_suffix(".preview.png"))
    views = [(15, -60), (15, 30), (15, 120), (60, -60)]
    fig = plt.figure(figsize=(4 * len(views), 4))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        # GLB is Y-up; matplotlib is Z-up.
        ax.plot_trisurf(v[:, 0], -v[:, 2], v[:, 1], triangles=mesh.faces,
                        color="#b9c4d6", edgecolor="none", shade=True, linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        lim = (-1, 1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
        ax.set_axis_off()
    fig.suptitle(f"{Path(a.glb).name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="queue an image for 3D generation")
    s.add_argument("image")
    s.add_argument("--model", choices=MODELS, default="v2",
                   help="shape model preset (default v2; turbo variants are ~5x faster)")
    s.add_argument("--steps", type=int, help="diffusion steps (default: 30, or 5 for turbo)")
    s.add_argument("--octree", type=int, default=256, help="mesh resolution: 256 fast, 380 detailed")
    s.add_argument("--guidance", type=float, default=5.0)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--texture", action="store_true", help="also paint a texture (slow, more VRAM)")
    s.add_argument("--max-faces", type=int, default=40000, help="decimate to this many faces (0 = keep)")
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
    pv.add_argument("--faces", type=int, default=15000, help="decimate for plotting speed")

    a = p.parse_args()
    handler = {"submit": cmd_submit, "wait": cmd_wait, "fetch": cmd_fetch, "status": cmd_status,
               "clean": cmd_clean, "reset-queue": cmd_reset_queue, "preview": cmd_preview}[a.cmd]
    sys.exit(handler(a) or 0)


if __name__ == "__main__":
    main()
