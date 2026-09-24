"""Which queries get their full wire published, and why.

The site's table covers all 323 test queries; the browsable examples cannot,
because the wire is 25,840 records of roughly 3 KB. So a subset is curated, and
the only thing worth arguing about is how it is chosen: a subset picked for how
well it reads would turn a negative result into a highlight reel.

The rule is quotas over outcomes, not over prettiness. Four buckets, each filled
best-first from the queries that belong in it, none of them padded when it runs
short. Ties break on the query id so the same run always produces the same page.
"""

# The failure bucket is filled first: 313 of Jev's 6,460 calls never succeeded
# and those passages kept their BM25 slot, which is the kind of thing a curated
# subset quietly loses.
QUOTAS = {"jev-failure": 2, "jev-gain": 8, "laya-loss": 8, "unchanged": 6}


def _sorted(queries, key, keep):
    return sorted((q for q in queries if keep(q)), key=lambda q: (key(q), q["query_id"]))


def select_examples(queries, total=None):
    """-> [{"query_id", "reason"}], best-first within each bucket.

    `queries` are dicts of query_id, jev_delta, laya_delta (per-query nDCG@10
    against the BM25 floor) and jev_failed (calls that never succeeded).
    `total` caps the whole subset; by default it is the sum of the quotas.
    """
    ranked = {
        "jev-failure": _sorted(queries, lambda q: -q["jev_failed"], lambda q: q["jev_failed"] > 0),
        "jev-gain": _sorted(queries, lambda q: -q["jev_delta"], lambda q: q["jev_delta"] > 0),
        "laya-loss": _sorted(queries, lambda q: q["laya_delta"], lambda q: q["laya_delta"] < 0),
        "unchanged": _sorted(queries, lambda q: 0, lambda q: q["jev_delta"] == 0 == q["laya_delta"]),
    }
    cap = sum(QUOTAS.values()) if total is None else total
    picked, taken = [], set()
    for reason, quota in QUOTAS.items():
        for q in ranked[reason]:
            if len(picked) == cap or quota == 0:
                break
            if q["query_id"] in taken:
                continue
            picked.append({"query_id": q["query_id"], "reason": reason})
            taken.add(q["query_id"])
            quota -= 1
    return picked
