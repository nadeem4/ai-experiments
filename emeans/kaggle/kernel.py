"""Embeds the arXiv-CS sample on a Kaggle GPU, once, untimed.

    kaggle kernels push -p emeans/kaggle

Clones this repository at main, samples the attached Cornell metadata with
`emeans.arxiv`, embeds it with `emeans.embed`, and leaves vectors.npy,
months.npy, ids.txt and meta.json in the notebook's working directory. No secrets.
"""
import glob
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nadeem4/ai-experiments.git"
CHECKOUT = Path("/kaggle/temp/ai-experiments")  # outside working/, so it is not fetched
OUT = Path("/kaggle/working")


def run(*command, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def main():
    import torch

    print(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("no GPU: turn the accelerator on for this notebook.", flush=True)

    run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, CHECKOUT)
    run("git", "-C", CHECKOUT, "sparse-checkout", "set", "emeans")
    run(sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "einops")
    sys.path.insert(0, str(CHECKOUT))

    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    from emeans import arxiv, embed

    meta = glob.glob("/kaggle/input/**/arxiv-metadata-oai-snapshot.json", recursive=True)
    if not meta:
        raise SystemExit("the Cornell arxiv dataset is not attached")
    sample = arxiv.sample(arxiv.read_jsonl(meta[0]))
    counts = arxiv.month_counts(sample)
    print(f"{len(sample):,} docs over {len(counts)} months, "
          f"{min(counts)} to {max(counts)}", flush=True)

    # Pin the exact commit actually used and record it.
    revision = model_info(embed.MODEL, revision=embed.REVISION).sha
    model = SentenceTransformer(embed.MODEL, trust_remote_code=True, revision=revision,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.half()
    embed.run(model, sample, OUT, revision=revision)
    shutil.rmtree(OUT / "shards", ignore_errors=True)  # shards only matter for resuming

    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}  {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
