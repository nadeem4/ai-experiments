"""Thread pinning and the environment every run records."""
import datetime
import importlib.metadata
import os
import platform
import subprocess

import faiss
from threadpoolctl import threadpool_limits

_limits = None  # held so the limit stays in force for the life of the process


def physical_cores():
    import psutil

    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def pin_threads(n):
    """The same thread count for faiss and for every BLAS/OpenMP pool."""
    global _limits
    faiss.omp_set_num_threads(n)
    _limits = threadpool_limits(limits=n)


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    return {
        "faiss": _version("faiss-cpu"),
        "scikit-learn": _version("scikit-learn"),
        "numpy": _version("numpy"),
        "sentence-transformers": _version("sentence-transformers"),
        "cpu": _cpu(),
        "threads": faiss.omp_get_max_threads(),
        "commit": _commit(),
        "date": datetime.date.today().isoformat(),
    }
