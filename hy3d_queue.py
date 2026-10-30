"""Git-backed job queue shared by the client (Claude's sandbox) and the Kaggle worker.

Jobs live on an orphan branch (default ``queue``) of this repo:

    jobs/<job_id>/job.json      written by the client (image name + generation options)
    jobs/<job_id>/input.<ext>   the source image
    jobs/<job_id>/status.json   written by the worker (claimed / running / done / error)
    jobs/<job_id>/result.glb    written by the worker
    worker/heartbeat.json       written by the worker while it is alive

The client and worker never write the same file, so concurrent pushes always
rebase cleanly. Every write goes through ``publish``: fetch, hard-reset to the
remote branch, copy files in, commit, push, retry on rejection. That also makes
both sides tolerate the branch being force-reset (see ``client reset-queue``).

Stdlib only, so it runs anywhere.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

BRANCH = os.environ.get("HY3D_QUEUE_BRANCH", "queue")


def git(repo, *args, check=True, capture=True):
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{res.stderr or res.stdout}")
    return res


def has_transparency(image, min_fraction=0.01):
    """True if a PIL image is really cut out: at least ``min_fraction`` of its pixels are transparent.

    Many PNGs carry an alpha channel that is opaque almost everywhere; treating any alpha < 255
    as "already cut out" skips background removal and feeds the white background to the model.
    """
    if "A" not in image.getbands():
        return False
    hist = image.getchannel("A").histogram()
    return sum(hist[:128]) >= min_fraction * image.width * image.height


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Queue:
    def __init__(self, workdir, remote_url, branch=BRANCH, name="hy3d", email="hy3d@localhost", config=None):
        self.dir = Path(workdir)
        self.config = config or {}
        self.url = remote_url
        self.branch = branch
        self.name = name
        self.email = email

    # -- setup -------------------------------------------------------------

    def ensure(self):
        """Clone (or create) the queue branch into self.dir."""
        if not (self.dir / ".git").exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            git(self.dir, "init", "-q")
            git(self.dir, "remote", "add", "origin", self.url)
        else:
            git(self.dir, "remote", "set-url", "origin", self.url)
        git(self.dir, "config", "user.name", self.name)
        git(self.dir, "config", "user.email", self.email)
        for k, v in self.config.items():
            git(self.dir, "config", k, v)
        if not self._fetch():
            # Branch doesn't exist on the remote yet: create it as an orphan.
            git(self.dir, "checkout", "-q", "--orphan", self.branch)
            git(self.dir, "rm", "-rqf", "--ignore-unmatch", ".")
            (self.dir / "README.md").write_text(
                "Job queue for hy3d. Managed by hy3d_queue.py; do not edit by hand.\n"
            )
            git(self.dir, "add", "-A")
            git(self.dir, "commit", "-qm", "init queue")
            res = git(self.dir, "push", "-q", "-u", "origin", self.branch, check=False)
            if res.returncode != 0 and not self._fetch():
                raise RuntimeError(res.stderr)
        self.sync()
        return self

    def _fetch(self):
        res = git(self.dir, "fetch", "-q", "origin", self.branch, check=False)
        return res.returncode == 0

    def sync(self):
        """Make the working tree exactly match the remote branch."""
        if not self._fetch():
            raise RuntimeError(f"cannot fetch branch {self.branch!r} from {self.url}")
        git(self.dir, "checkout", "-q", "-B", self.branch, f"origin/{self.branch}")
        git(self.dir, "reset", "-q", "--hard", f"origin/{self.branch}")
        git(self.dir, "clean", "-qfd")

    # -- writes ------------------------------------------------------------

    def publish(self, files, message, remove=(), retries=8):
        """Write ``files`` ({relpath: bytes|str|Path}) and delete ``remove`` paths, then push.

        Retries on push rejection by re-syncing and re-applying the change.
        """
        for attempt in range(retries):
            self.sync()
            for rel in remove:
                p = self.dir / rel
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            for rel, content in files.items():
                p = self.dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, Path):
                    shutil.copyfile(content, p)
                elif isinstance(content, bytes):
                    p.write_bytes(content)
                else:
                    p.write_text(content)
            git(self.dir, "add", "-A")
            if git(self.dir, "diff", "--cached", "--quiet", check=False).returncode == 0:
                return False  # nothing changed
            git(self.dir, "commit", "-qm", message)
            if git(self.dir, "push", "-q", "origin", self.branch, check=False).returncode == 0:
                return True
            time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"push to {self.branch} kept failing after {retries} attempts")

    # -- reads -------------------------------------------------------------

    def jobs(self):
        root = self.dir / "jobs"
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "job.json").exists())

    def read_json(self, rel):
        p = self.dir / rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def job(self, job_id):
        return self.read_json(f"jobs/{job_id}/job.json")

    def status(self, job_id):
        return self.read_json(f"jobs/{job_id}/status.json")
