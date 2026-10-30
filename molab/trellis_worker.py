import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # TRELLIS.2 worker

        Serves image-to-3D jobs from the `queue` branch of the GitHub repo (submitted with
        `python3 hy3d.py submit <image>`), using TRELLIS.2 on this notebook's GPU.

        1. Turn on the GPU: the **specs button** in the header → GPU (RTX PRO 6000 Blackwell).
        2. **GitHub token**: fine-grained, Contents read and write on the repo.
        3. **Hugging Face token**: TRELLIS.2 uses Meta's DINOv3, which is gated. Accept the license at
           [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
           then create a *read* token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
        4. Press **Start**. First setup takes ~10 min (longer if it has to compile); then it serves jobs until stopped.

        Tokens typed here stay in this session's memory; they are not saved into the notebook.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    repo = mo.ui.text(value="Eslezer/3DENGINE", label="Repo (owner/name)")
    gh_token = mo.ui.text(kind="password", label="GitHub token", full_width=True)
    hf_token = mo.ui.text(kind="password", label="Hugging Face token", full_width=True)
    start = mo.ui.run_button(label="Start worker")
    mo.vstack([repo, gh_token, hf_token, start])
    return gh_token, hf_token, repo, start


@app.cell
def _(gh_token, hf_token, mo, repo, start):
    import base64
    import os
    import subprocess
    import sys

    mo.stop(not start.value, mo.md("Fill in both tokens, then press **Start worker**."))
    mo.stop(not gh_token.value or not hf_token.value, mo.md("**Both tokens are required.**"))

    def run(cmd, cwd=None, env=None):
        # Relay output line by line so progress shows up while the cell runs.
        print("+", " ".join(c for c in cmd if "extraheader" not in c), flush=True)
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
        if p.wait():
            raise RuntimeError(f"command failed with exit code {p.returncode}")

    env = dict(os.environ, GITHUB_TOKEN=gh_token.value, HF_TOKEN=hf_token.value, HY3D_ENGINE="trellis2")
    auth = base64.b64encode(f"x-access-token:{gh_token.value}".encode()).decode()
    hdr = ["-c", f"http.extraheader=AUTHORIZATION: basic {auth}"]
    src = "/tmp/hy3d/3DENGINE"
    if os.path.exists(src):
        run(["git", *hdr, "-C", src, "pull", "-q"])
    else:
        run(["git", *hdr, "clone", "-q", "--depth", "1", "-b", "main",
             f"https://github.com/{repo.value}.git", src])

    run(["nvidia-smi"])
    run([sys.executable, "worker.py", "--engine", "trellis2", "--setup-only", "--repo", repo.value],
        cwd=src, env=env)
    run([sys.executable, "-u", "worker.py", "--engine", "trellis2", "--repo", repo.value], cwd=src, env=env)
    return


if __name__ == "__main__":
    app.run()
