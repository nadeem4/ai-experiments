"""Runs the re-ranking experiment on a Kaggle GPU.

    kaggle kernels push -p rerank/kaggle

Clones this repository at main, installs what the Kaggle image lacks, and runs
`exp run rerank` with its output pointed at the notebook's working directory.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nadeem4/ai-experiments.git"
CHECKOUT = Path("/kaggle/working/ai-experiments")
OUT = Path("/kaggle/working/rerank")
METHODS = ["bm25", "laya-score", "laya-typed-score", "jev-score", "cross-encoder"]
# This account's other notebooks have their own secrets; this one is read into
# OPENROUTER_API_KEY, which is where the reranker looks.
SECRET = "AI_EXPERIMENTS_OPENROUTER"


def run(*command, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def load_key():
    """Kaggle's secret store into the environment, where the reranker looks."""
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret(SECRET)
        print(f"{SECRET}: loaded", flush=True)
        return True
    except Exception as e:
        print(f"{SECRET}: not available ({type(e).__name__})", flush=True)
        return False


def main():
    import torch

    print(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"{torch.cuda.get_device_name(0)}, "
              f"capability {torch.cuda.get_device_capability(0)}", flush=True)
    else:
        print("no GPU: turn the accelerator on for this notebook.", flush=True)

    run("git", "clone", "--depth", "1", REPO, CHECKOUT)
    run(sys.executable, "-m", "pip", "install", "-q",
        "laya>=0.3.20", "rank-bm25>=0.2.2", "pytrec-eval-terrier>=0.5.7",
        "sentence-transformers>=5.0", "huggingface-hub>=1.0")

    # Without the key, jev-score is dropped rather than run: a failed call is
    # recorded like any other, so 6,460 of them would be skipped as already done
    # by the run that has the key.
    methods = METHODS if load_key() else [m for m in METHODS if not m.startswith("jev")]
    if len(methods) < len(METHODS):
        print("running without jev-score. Attach the secret and push again.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "-m", "exp", "run", "rerank",
        "--tag", "gpu", "--out", OUT, "--limit", "0", "--top-k", "20",
        "--methods", *methods, cwd=CHECKOUT)

    print("\nwhat came out:", flush=True)
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(OUT)}  {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
