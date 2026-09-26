"""Submitting an experiment to a Kaggle notebook, and checking it is worth doing.

The kernel clones this repository at `main` and runs what it finds there. So the
prerequisites are mostly about git: uncommitted work, unpushed commits and the
wrong branch all produce a run that looks entirely normal and measures code that
is not the code in front of you.

Two prerequisites cannot be checked from here -- whether the notebook has its
secret attached, and whether its accelerator is on -- so they are printed as
steps to do rather than assumed.
"""
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH = "main"


def kernel_dir(name):
    return REPO_ROOT / name / "kaggle"


def slug(directory):
    """The notebook this experiment pushes to, read from its own metadata."""
    path = Path(directory) / "kernel-metadata.json"
    if not path.exists():
        raise SystemExit(f"no kernel-metadata.json in {directory}. This experiment "
                         f"has no Kaggle notebook to push to.")
    return json.loads(path.read_text(encoding="utf-8"))["id"]


def git_state():
    """-> {dirty, unpushed, branch}, as the submit checks need it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()

    dirty = [line[2:].lstrip() for line in git("status", "--porcelain").splitlines() if line]
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    unpushed = git("rev-list", "--count", f"origin/{BRANCH}..HEAD")
    return {"dirty": dirty, "branch": branch,
            "unpushed": int(unpushed) if unpushed.isdigit() else 0}


def authenticated():
    out = subprocess.run(["uv", "run", "kaggle", "config", "view"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    return "username:" in out.stdout


def problems(git, metadata, authenticated):
    """-> every reason this submission would be wrong, not just the first.

    All of them at once: finding out about the second one after fixing the first
    costs another round trip, and the whole point is to fail before the run
    rather than 40 minutes into it."""
    found = []
    if git["dirty"]:
        found.append(f"uncommitted: {', '.join(git['dirty'][:5])}"
                     f"{' and more' if len(git['dirty']) > 5 else ''}. The kernel "
                     f"does a fresh clone, so these changes would not be in the run.")
    if git["unpushed"]:
        found.append(f"{git['unpushed']} commit(s) not pushed. The kernel clones "
                     f"from the remote: git push first.")
    if git["branch"] != BRANCH:
        found.append(f"on branch {git['branch']}, but the kernel clones {BRANCH}. "
                     f"The run would measure {BRANCH}, not what you are looking at.")
    if not metadata:
        found.append("no kernel-metadata.json for this experiment.")
    if not authenticated:
        found.append("the kaggle CLI is not authenticated: run "
                     "`uv run kaggle auth login`.")
    return found


def manual_steps(notebook):
    """What has to be true on Kaggle's side, which no API here can confirm."""
    url = f"https://www.kaggle.com/code/{notebook}/edit"
    return [
        f"the notebook's accelerator is a GPU: {url} -> Session options",
        f"OPENROUTER_API_KEY is attached: {url} -> Add-ons -> Secrets",
    ]


def submit(name, dry_run=False):
    directory = kernel_dir(name)
    notebook = slug(directory)
    found = problems(git_state(), (directory / "kernel-metadata.json").exists(), authenticated())

    if found:
        print(f"not submitting {name}:")
        for problem in found:
            print(f"  - {problem}")
        return 1

    print(f"{name} -> {notebook}")
    print("check on Kaggle, which cannot be checked from here:")
    for step in manual_steps(notebook):
        print(f"  - {step}")
    if dry_run:
        print("\ndry run: not pushed.")
        return 0

    subprocess.run(["uv", "run", "kaggle", "kernels", "push", "-p", str(directory)],
                   cwd=REPO_ROOT, check=True)
    print(f"\nwatch:  uv run cli status {name}")
    print(f"fetch:  uv run cli fetch {name}")
    return 0


def status(name):
    subprocess.run(["uv", "run", "kaggle", "kernels", "status", slug(kernel_dir(name))],
                   cwd=REPO_ROOT, check=True)
    return 0


def fetch(name, into):
    into = Path(into or REPO_ROOT / name / "runs" / "kaggle")
    into.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uv", "run", "kaggle", "kernels", "output",
                    slug(kernel_dir(name)), "-p", str(into)], cwd=REPO_ROOT, check=True)
    print(f"\ninto {into}")
    return 0
