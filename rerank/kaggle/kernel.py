"""Run the re-ranking experiment on a Kaggle GPU.

    kaggle kernels push -p rerank/kaggle

This machine has no usable GPU, so every local model in this repository has been
measured on CPU and the write-up says in those words that nothing in it is a
speed claim. A Kaggle notebook has a T4 for free, which is enough to measure the
whole thing on one machine, on one device, in one run.

## What this does NOT do

It does not resume a local run or merge with one. The wire log written here is a
new measurement on different hardware: laya runs in float16 on CUDA and float32
on CPU, so its answers are not the ones the CPU produced, and mixing them would
be comparing two things while calling them one. The results land under their own
tag and `usable_device()` records `cuda` on every row.

## The key

The Jev pass is the only method that leaves the machine. Its key comes from
Kaggle's own secret store, attached to this notebook through Kaggle's UI, and is
put into the environment where `rerank.rerankers.jev` already looks for it. It is
never written to a file, never printed, and never appears in the output.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nadeem4/ai-experiments.git"
CHECKOUT = Path("/kaggle/working/ai-experiments")
# Everything the run produces goes straight into the notebook's output, which is
# what `kaggle kernels output` downloads: the committed results AND the raw wire
# log, because the log is the evidence the results are derived from.
OUT = Path("/kaggle/working/rerank")

METHODS = ["bm25", "laya-score", "laya-typed-score", "jev-score", "cross-encoder"]


def run(*command, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def load_key():
    """Read OPENROUTER_API_KEY out of Kaggle's secret store into the environment.

    Absent, the four local methods still run and only the Jev pass fails, which
    is a better outcome than refusing to start: the GPU work is the reason to be
    here at all."""
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
        print("OPENROUTER_API_KEY: loaded from Kaggle secrets", flush=True)
    except Exception as e:
        print(f"OPENROUTER_API_KEY: NOT available ({type(e).__name__}). "
              f"The local methods will run; jev-score will fail every call.", flush=True)


def main():
    import torch

    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}, "
              f"capability {torch.cuda.get_device_capability(0)}", flush=True)
    else:
        print("NO GPU. Turn the accelerator on for this notebook; on CPU this "
              "takes hours and is the measurement we already have.", flush=True)

    run("git", "clone", "--depth", "1", REPO, CHECKOUT)
    # The experiment's own dependencies. torch, matplotlib and pyarrow ship with
    # the Kaggle image; these are the ones that do not.
    run(sys.executable, "-m", "pip", "install", "-q",
        "laya>=0.3.20", "rank-bm25>=0.2.2", "pytrec-eval-terrier>=0.5.7",
        "sentence-transformers>=5.0", "huggingface-hub>=1.0")

    load_key()
    OUT.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "-m", "exp", "run", "rerank",
        "--tag", "gpu", "--out", OUT, "--limit", "0", "--top-k", "20",
        "--methods", *METHODS, cwd=CHECKOUT)

    print("\nwhat came out:", flush=True)
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(OUT)}  {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
