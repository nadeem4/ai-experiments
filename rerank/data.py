"""BEIR NFCorpus, official test split, from the Hugging Face mirror of BEIR's
own files: BeIR/nfcorpus (corpus + queries, parquet) and BeIR/nfcorpus-qrels
(train/dev/test.tsv, the official judgements).

Why these files and not the `beir` package: `beir` pulls sentence-transformers,
elasticsearch and a torch stack in just to unzip a dataset, and it fetches a zip
from a university host that is regularly down. `ir_datasets` would also work.
hf_hub_download gets the identical official files, caches them, resumes a broken
download, and adds one small dependency instead of a framework.
"""
import csv
import io
from pathlib import Path

CORPUS_REPO = "BeIR/nfcorpus"
QRELS_REPO = "BeIR/nfcorpus-qrels"


def parse_qrels(text):
    """The TREC-style qrels TSV -> {query_id: {doc_id: grade}}. NFCorpus grades
    are 0, 1 or 2; the metrics treat anything above 0 as relevant."""
    rows = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(rows, None)
    qrels = {}
    for i, row in enumerate(rows, start=2):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) < 3:
            raise ValueError(f"qrels line {i} has {len(row)} fields, expected 3: {row!r}")
        qid, doc_id, grade = row[0], row[1], row[2]
        qrels.setdefault(qid, {})[doc_id] = int(grade)
    return qrels


def _parquet(repo, filename, cache_dir):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", cache_dir=cache_dir)
    return pq.read_table(path).to_pylist()


def load_nfcorpus(cache_dir=None, split="test"):
    """-> (corpus, queries, qrels). Only queries with judgements are returned, so
    the query set is exactly the official split (323 queries for test)."""
    from huggingface_hub import hf_hub_download
    cache_dir = str(cache_dir) if cache_dir else None

    qrels_path = hf_hub_download(repo_id=QRELS_REPO, filename=f"{split}.tsv", repo_type="dataset", cache_dir=cache_dir)
    qrels = parse_qrels(Path(qrels_path).read_text(encoding="utf-8"))

    corpus = {row["_id"]: {"title": row.get("title") or "", "text": row.get("text") or ""}
              for row in _parquet(CORPUS_REPO, "corpus/corpus-00000-of-00001.parquet", cache_dir)}
    queries = {row["_id"]: row["text"] for row in _parquet(CORPUS_REPO, "queries/queries-00000-of-00001.parquet", cache_dir)
               if row["_id"] in qrels}
    return corpus, queries, qrels
