# hy3d worker for Kaggle: plain-Python version of hy3d_worker.ipynb.
#
# Use it either way:
#   * paste the whole file into a single cell of a new Kaggle notebook, or
#   * Kaggle "New Notebook" -> File -> Import Notebook -> upload this .py.
#
# Before running: Settings -> Accelerator: GPU T4 x2, Settings -> Internet: on,
# Add-ons -> Secrets: GITHUB_TOKEN (fine-grained token, Contents: read and write on the repo).
# It runs until you stop it. Setup takes ~15 min the first time in each session.

REPO = "Eslezer/3DENGINE"  # owner/name of the repo holding the queue branch
TEXTURE = True             # build Hunyuan texture painting; False = shapes only (faster setup)

import base64
import os
import subprocess
import sys

from kaggle_secrets import UserSecretsClient


def run(cmd, cwd=None):
    # Relay output line by line: in a notebook cell, a child's stdout would otherwise
    # go to the kernel log instead of the cell.
    print("+", " ".join(c for c in cmd if not c.startswith("http.extraheader")), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"command failed with exit code {p.returncode}")


os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
auth = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
hdr = ["-c", f"http.extraheader=AUTHORIZATION: basic {auth}"]
SRC = "/tmp/hy3d/3DENGINE"  # outside /kaggle/working so nothing here lands in notebook output
if os.path.exists(SRC):
    run(["git", *hdr, "-C", SRC, "pull", "-q"])
else:
    run(["git", *hdr, "clone", "-q", "--depth", "1", "-b", "main",
         f"https://github.com/{REPO}.git", SRC])

run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
run([sys.executable, "worker.py", "--setup-only"] + (["--texture"] if TEXTURE else []), cwd=SRC)
# Serves jobs forever. Model weights download from Hugging Face on the first job.
run([sys.executable, "-u", "worker.py", "--repo", REPO], cwd=SRC)
