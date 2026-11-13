"""The growing index on a short synthetic timeline: 2019-12 is the initial
period, then five arrival months of 40 docs with 5 held out as queries."""
import numpy as np
import pytest

from emeans import growing, index

MONTHS = [f"2020-{m:02d}" for m in range(1, 6)]
N_Q = 5


def month_ids():
    return np.array(["2019-12"] * 30 + [m for m in MONTHS for _ in range(40)])


def test_timeline_is_2020_01_onwards_and_72_at_full_scale():
    t = growing.timeline(72)
    assert len(t) == 72 and t[0] == "2020-01" and t[-1] == "2025-12"
    assert growing.timeline(3) == ["2020-01", "2020-02", "2020-03"]
    assert growing.INITIAL == ("2018-01", "2019-12")


def test_held_out_queries_are_fixed_and_do_not_depend_on_the_run_seed():
    a = growing.hold_out(month_ids(), n=N_Q, dataset_seed=0)
    b = growing.hold_out(month_ids(), n=N_Q, dataset_seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a[0]) == N_Q * len(MONTHS)
    assert not np.array_equal(a[0], growing.hold_out(month_ids(), n=N_Q, dataset_seed=1)[0])


def test_queries_and_indexed_are_disjoint_and_cover_everything():
    queries, indexed = growing.hold_out(month_ids(), n=N_Q)
    assert not set(queries) & set(indexed)
    assert sorted(set(queries) | set(indexed)) == list(range(len(month_ids())))


def test_a_query_is_in_no_indexed_prefix_initial_set_or_training_set():
    """Review Focus 2: leakage pin."""
    ids = month_ids()
    queries, indexed = growing.hold_out(ids, n=N_Q)
    indexed_months = ids[indexed]
    initial = indexed[:growing.n_through(indexed_months, "2019-12")]
    assert not set(queries) & set(initial)
    for month in MONTHS:
        rows = indexed[:growing.n_through(indexed_months, month)]
        assert not set(queries) & set(rows)
    qs = growing.make_split(np.arange(len(ids), dtype=np.float32)[:, None], ids, 5, n_queries=N_Q)
    seen = {int(v) for v in qs.vectors[:, 0]}
    held = {int(v) for q in qs.queries.values() for v in q[:, 0]}
    assert not seen & held and len(held) == N_Q * 5


def test_indexed_is_chronological_and_prefixes_grow_by_one_months_non_queries():
    ids = month_ids()
    _, indexed = growing.hold_out(ids, n=N_Q)
    months = ids[indexed]
    assert list(months) == sorted(months)
    sizes = [growing.n_through(months, m) for m in ["2019-12"] + MONTHS]
    assert sizes[0] == 30 and np.diff(sizes).tolist() == [40 - N_Q] * 5


def test_a_month_with_too_few_docs_is_named():
    ids = np.array(["2019-12"] * 10 + ["2020-01"] * 3 + ["2020-02"] * 40)
    with pytest.raises(ValueError, match="2020-01"):
        growing.hold_out(ids, n=N_Q)


# ---- the four maintenance policies (3 drifting blobs, k = 6) ----

K = 6


@pytest.fixture(autouse=True)
def one_thread():
    from emeans import env
    before = env.faiss.omp_get_max_threads()
    env.pin_threads(1)
    yield
    env.pin_threads(before)


def blobs(n, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 10, 0, 0]], dtype=np.float32) + drift
    return (centres[rng.integers(0, 3, n)] + rng.normal(0, 1, (n, 4))).astype(np.float32)


@pytest.fixture(scope="module")
def start():
    x0 = blobs(900)
    return x0, growing.initial_centroids(x0, K, seed=0)


def step(arm, start, month="2020-02", seed=0, centroids=None):
    x0, c0 = start
    x = np.concatenate([x0, blobs(300, drift=1.0, seed=1)])
    labels_prev = index.reassign(x0, c0)[0]
    return growing.maintain(arm, c0 if centroids is None else centroids, x, month, seed, labels_prev)


def test_stale_keeps_the_centroids_and_costs_nothing(start):
    m = step("stale", start)
    assert np.array_equal(m.centroids, start[1])
    assert m.train_seconds == 0 and m.assign_seconds == 0 and m.rows == 0
    assert len(m.labels) == 1200 and m.labels.min() >= 0


def test_yearly_changes_centroids_only_in_january_and_ignores_the_old_ones(start):
    assert np.array_equal(step("yearly", start, "2020-02").centroids, start[1])
    jan = step("yearly", start, "2021-01")
    assert not np.array_equal(jan.centroids, start[1])
    other = step("yearly", start, "2021-01", centroids=start[1] + 5)
    assert np.array_equal(jan.centroids, other.centroids)  # a rebuild, not a warm start
    assert jan.train_seconds > 0 and jan.assign_seconds > 0


def test_emeans_is_warm_started_not_cold(start):
    a = step("emeans", start)
    b = step("emeans", start, centroids=start[1] + 3)
    assert not np.array_equal(a.centroids, b.centroids)  # the given centroids matter
    x0 = start[0]
    x = np.concatenate([x0, blobs(300, drift=1.0, seed=1)])
    from emeans import arms
    cold = arms.train_emeans(x, K, 0.5, arms.random_init(x, K, 0), seed=growing.seed_for(0, "2020-02"))
    assert not np.array_equal(a.centroids, cold.centroids)


def test_emeans_and_lloyd_half_touch_the_same_rows_within_one_batch(start):
    """Review Focus 3."""
    e, h = step("emeans", start), step("lloyd_half", start)
    assert e.rows == pytest.approx(1200 / 2, abs=4) and h.rows == 600
    assert abs(e.rows - h.rows) <= 4  # one e-means batch at this size


def test_no_label_is_minus_one_after_a_half_epoch_of_emeans(start):
    """Review Focus 3: labels come from our own re-assignment, not em.assignments."""
    for arm in growing.ARMS_B:
        m = step(arm, start, "2021-01")
        assert m.labels.min() >= 0 and m.labels.max() < K and len(m.labels) == 1200


def test_policies_are_deterministic_for_a_seed(start):
    for arm in ("yearly", "emeans", "lloyd_half"):
        a, b = step(arm, start, "2021-01"), step(arm, start, "2021-01")
        assert np.array_equal(a.centroids, b.centroids), arm


def test_arms_start_from_identical_initial_centroids(start):
    again = growing.initial_centroids(start[0], K, seed=0)
    assert np.array_equal(again, start[1])


# ---- the month loop, records and ledger (5 months, k = 8, 2 seeds) ----

import json  # noqa: E402

from emeans import store, tables  # noqa: E402
from emeans.experiment import Scale  # noqa: E402

SCALE = Scale(n=None, n_test=0, n_queries=10, k=8, seeds_a=1, seeds_bc=2, b_months=5, nprobes=(2, 8))
VOLATILE = {"train_seconds", "assign_seconds", "maintenance_seconds", "adding_seconds", "eval_seconds",
            "search", "observed_threads"}


@pytest.fixture(scope="module")
def gsplit():
    rng = np.random.default_rng(3)
    vectors = np.concatenate([blobs(400, seed=10)] + [blobs(100, drift=i * 0.5, seed=11 + i) for i in range(5)])
    ids = np.array(["2019-12"] * 400 + [m for m in MONTHS for _ in range(100)])
    return growing.make_split(vectors, ids, 5, n_queries=10)


def run_b(gs, path, scale=SCALE):
    growing.run(gs, scale, path, seeds=range(2), nprobes=(2, 8), threads=1, log=lambda *_: None)
    return store.load(path)


def stable(records):
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in records]


def test_units_are_chains_of_months_arm_by_arm_seed_by_seed():
    us = growing.units(SCALE)
    assert len(us) == 4 * 2 * 5 and len(set(us)) == len(us)
    assert us[0] == ("b", "arxiv", "stale", "2020-01", 0) and us[5] == ("b", "arxiv", "yearly", "2020-01", 0)


def test_one_complete_record_per_arm_seed_month(gsplit, tmp_path):
    records = run_b(gsplit, tmp_path / "b.jsonl")
    assert sorted(map(store.key, records)) == sorted(growing.units(SCALE))
    for r in records:
        assert not [f for f in growing.REQUIRED if f not in r], r["arm"]
        assert r["maintenance_seconds"] == r["train_seconds"] + r["assign_seconds"]
        assert set(r["search"]) == {"2", "8"} and len(r["recall_per_query"]["8"]) == 10
        assert r["search"]["8"]["recall10"] == 1.0  # nprobe == k scans everything
    json.dumps(records)


def test_rerunning_appends_nothing(gsplit, tmp_path):
    path = tmp_path / "b.jsonl"
    run_b(gsplit, path)
    before = path.read_bytes()
    run_b(gsplit, path)
    assert path.read_bytes() == before


def test_interrupting_after_month_3_gives_the_same_records_as_an_uninterrupted_run(gsplit, tmp_path):
    whole = run_b(gsplit, tmp_path / "whole" / "b.jsonl")
    part = tmp_path / "part" / "b.jsonl"
    run_b(gsplit, part, SCALE._replace(b_months=3))
    assert len(store.load(part)) == 4 * 2 * 3
    resumed = run_b(gsplit, part)
    key = lambda r: store.key(r)  # noqa: E731
    assert stable(sorted(resumed, key=key)) == stable(sorted(whole, key=key))


def test_a_crash_between_state_and_record_replays_that_month(gsplit, tmp_path):
    whole = run_b(gsplit, tmp_path / "whole" / "b.jsonl")
    path = tmp_path / "crash" / "b.jsonl"
    run_b(gsplit, path)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))  # the last record never landed, its state did
    key = lambda r: store.key(r)  # noqa: E731
    assert stable(sorted(run_b(gsplit, path), key=key)) == stable(sorted(whole, key=key))


def test_a_corrupt_state_file_exits_naming_the_arm_and_seed(gsplit, tmp_path):
    path = tmp_path / "b.jsonl"
    run_b(gsplit, path, SCALE._replace(b_months=3))
    (tmp_path / "state" / "emeans-1-2020-03.npz").write_bytes(b"garbage")
    with pytest.raises(SystemExit, match="emeans.*seed 1"):
        run_b(gsplit, path)


def test_a_state_that_does_not_match_the_last_record_exits(gsplit, tmp_path):
    path = tmp_path / "b.jsonl"
    run_b(gsplit, path, SCALE._replace(b_months=3))
    f = tmp_path / "state" / "yearly-0-2020-03.npz"
    z = dict(np.load(f))
    z["centroids"] = z["centroids"] + 1
    np.savez(f, **z)
    with pytest.raises(SystemExit, match="yearly.*seed 0"):
        run_b(gsplit, path)


def test_every_arm_and_seed_shares_one_truth_file_per_month(gsplit, tmp_path):
    run_b(gsplit, tmp_path / "b.jsonl")
    assert sorted(p.stem for p in (tmp_path / "truth").glob("*.npy")) == MONTHS


def test_ledger_counts_train_plus_assign_never_adding_and_stale_is_zero(gsplit, tmp_path):
    """Review Focus 4."""
    records = run_b(gsplit, tmp_path / "b.jsonl")
    led = tables.ledger(records)
    assert set(led) == set(growing.ARMS_B)
    assert all(m == 0 and c == 0 for _, m, c in led["stale"]) and len(led["stale"]) == 5
    for arm, rows in led.items():
        assert [r[0] for r in rows] == MONTHS
        mine = [r for r in records if r["arm"] == arm]
        for month, monthly, cumulative in rows:
            per_seed = [r["train_seconds"] + r["assign_seconds"] for r in mine if r["setting"] == month]
            assert monthly == pytest.approx(sum(per_seed) / len(per_seed))
        assert [c for *_, c in rows] == pytest.approx(np.cumsum([m for _, m, _ in rows]).tolist())
    assert led["emeans"][-1][2] > 0 and led["yearly"][0][1] > 0  # January rebuilds in 2020-01
    assert led["yearly"][1][1] == 0  # no rebuild in February
    adding = tables.adding_seconds(records)
    assert set(adding) == set(growing.ARMS_B) and all(v > 0 for v in adding.values())
