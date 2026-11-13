"""Sampling the Cornell arXiv metadata (CC0): cs-primary papers, a uniform
sample per publish month.

Streams the multi-GB JSONL and keeps only the `per_month` records with the
smallest seeded hash of their id in each month. That is a uniform sample without
replacement, the same for the same seed, in bounded memory. No faiss or h5py
here: the Kaggle kernel imports this module.
"""
import hashlib
import heapq
import json
from email.utils import parsedate_to_datetime


def is_cs_primary(record):
    """The first category is the primary one; a cs cross-list does not count."""
    return record["categories"].split()[0].startswith("cs.")


def month_of(record):
    """`YYYY-MM` of the first version's date (RFC 2822 in the Cornell file)."""
    return parsedate_to_datetime(record["versions"][0]["created"]).strftime("%Y-%m")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _rank(seed, record_id):
    return hashlib.blake2b(f"{seed}:{record_id}".encode(), digest_size=8).digest()


def sample(records, per_month=10_000, seed=0, start="2018-01", end="2025-12"):
    """-> [{id, month, abstract}] sorted by month then id."""
    kept = {}  # month -> heap of (-rank, id, abstract): the largest rank is evicted first
    for r in records:
        if not is_cs_primary(r):
            continue
        month = month_of(r)
        if not start <= month <= end:
            continue
        heap = kept.setdefault(month, [])
        item = (_negate(_rank(seed, r["id"])), r["id"], r["abstract"].strip())
        if len(heap) < per_month:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    rows = [{"id": i, "month": m, "abstract": a} for m, heap in kept.items() for _, i, a in heap]
    return sorted(rows, key=lambda r: (r["month"], r["id"]))


def _negate(rank):
    """Bytes that sort in reverse, so the heap root is the worst-ranked kept record."""
    return bytes(255 - b for b in rank)


def month_counts(rows):
    counts = {}
    for r in rows:
        counts[r["month"]] = counts.get(r["month"], 0) + 1
    return dict(sorted(counts.items()))
