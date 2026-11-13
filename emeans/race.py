"""The resumable race runner shared by parts A and C.

One record per unit of work `(part, dataset, arm, setting, seed)`, appended as
it finishes; a unit already in the file is not run again. Calibration units come
first so budget units can read seconds-per-epoch back from the store.
"""
import hashlib
import time

from threadpoolctl import threadpool_info

from . import arms as arms_mod
from . import budgets, index, metrics, store
from .consts import ARMS, BUDGETS_S, EPOCHS, LLOYD_AT

CORE = ("part", "dataset", "arm", "setting", "seed", "n", "k", "d", "threads", "observed_threads")
EVAL = ("eval_seconds",)  # everything after training: untimed for comparisons, needed to project
TRAINED = ("centroids_hash", "train_seconds", "epochs", "rows")
QUALITY = ("test_mse", "imbalance", "dead_clusters")


def units(dataset, arm_names=ARMS, seeds=1, with_budgets=False, part="a"):
    """Calibration units, then per seed Lloyd `snapshots` and `iters=25`, the
    other arms at each of EPOCHS, and `budget=Ns` for every arm if asked."""
    out = [(part, dataset, arm, "calibration", 0) for arm in arm_names]
    for seed in range(seeds):
        if "lloyd" in arm_names:
            out += [(part, dataset, "lloyd", "snapshots", seed), (part, dataset, "lloyd", "iters=25", seed)]
        out += [(part, dataset, arm, f"epochs={e}", seed)
                for arm in arm_names if arm != "lloyd" for e in EPOCHS]
        if with_budgets:
            out += [(part, dataset, arm, f"budget={b}s", seed) for arm in arm_names for b in BUDGETS_S]
    return out


def required_fields(setting, with_search=True):
    """The fields a record for `setting` must carry; `run_unit` writes them and
    the pilot checks them."""
    if setting == "snapshots":
        return CORE + EVAL + ("mse_by_iter", "train_mse_by_iter")
    if setting == "calibration":
        return CORE + TRAINED
    fields = CORE + EVAL + TRAINED + QUALITY
    if setting.startswith("budget="):
        fields += ("budget_s", "overshoot")
    if with_search and (setting.startswith("iters=") or setting.startswith("budget=")):
        fields += ("search",)
    return fields


def calibration_from(records, dataset):
    """-> {arm: seconds per epoch} from the dataset's calibration records."""
    return {r["arm"]: r["train_seconds"] / r["epochs"] for r in records
            if r["setting"] == "calibration" and r["dataset"] == dataset}


def _hash(centroids):
    return hashlib.sha256(centroids.tobytes()).hexdigest()[:16]


def _observed_threads(threads):
    observed = max((p["num_threads"] for p in threadpool_info()), default=1)
    if observed > threads:
        raise SystemExit(f"a thread pool runs {observed} threads but the run is pinned to {threads}; "
                         "call env.pin_threads first")
    return observed


def run_unit(unit, split, scale, calibration, threads, with_search=True):
    part, dataset, arm, setting, seed = unit
    x, k = split.train, scale.k
    record = {"part": part, "dataset": dataset, "arm": arm, "setting": setting, "seed": seed,
              "n": len(x), "k": k, "d": x.shape[1], "threads": threads,
              "observed_threads": _observed_threads(threads)}
    init = arms_mod.random_init(x, k, seed)

    if setting == "snapshots":
        snaps = arms_mod.lloyd_snapshots(x, k, init)
        start = time.perf_counter()
        record["mse_by_iter"] = {str(i): metrics.mse(split.test, snaps[i]) for i in LLOYD_AT}
        record["train_mse_by_iter"] = {str(i): metrics.mse(x, snaps[i]) for i in LLOYD_AT}
        record["eval_seconds"] = time.perf_counter() - start
        return record

    if setting == "calibration":
        epochs = 1
    elif setting.startswith("epochs=") or setting.startswith("iters="):
        epochs = int(setting.split("=")[1])
    elif setting.startswith("budget="):
        if arm not in calibration:
            raise SystemExit(f"no calibration record for arm {arm!r} on {dataset!r}; "
                             "run the calibration units first")
        budget = int(setting[len("budget="):-1])
        epochs = budgets.epochs_for_budget(calibration[arm], budget)
    else:
        raise ValueError(f"unknown setting {setting!r}")

    trained = arms_mod.train(arm, x, k, epochs, init, seed)
    record.update(centroids_hash=_hash(trained.centroids), train_seconds=trained.seconds,
                  epochs=epochs, rows=trained.rows)
    if setting.startswith("budget="):
        record.update(budget_s=budget, overshoot=budgets.overshoot(trained.seconds, budget))
    if setting == "calibration":
        return record

    start = time.perf_counter()
    labels, _ = index.reassign(split.index, trained.centroids)
    record.update(test_mse=metrics.mse(split.test, trained.centroids),
                  imbalance=metrics.imbalance(labels, k),
                  dead_clusters=metrics.dead_clusters(labels, k))
    if with_search and (setting.startswith("iters=") or setting.startswith("budget=")):
        ivf = index.build_ivf(trained.centroids, split.index, labels)
        summary, _ = index.evaluate(ivf, split.queries, split.truth, scale.nprobes, base=split.index)
        record["search"] = {str(p): v for p, v in summary.items()}
    record["eval_seconds"] = time.perf_counter() - start
    return record


def run(unit_list, split, scale, store_path, threads=None, with_search=True, log=print):
    """Run every unit not already in `store_path`, appending each record as it
    finishes. Threads are pinned here and recorded in every record."""
    from . import env

    threads = threads or env.physical_cores()
    env.pin_threads(threads)
    done = store.done_keys(store_path)
    todo = [u for u in unit_list if u not in done]
    log(f"{len(unit_list) - len(todo)} units done, {len(todo)} to run")
    for i, unit in enumerate(todo, 1):
        calibration = calibration_from(store.load(store_path), unit[1])
        record = run_unit(unit, split, scale, calibration, threads, with_search)
        store.append(store_path, record)
        log(f"[{i}/{len(todo)}] {' '.join(map(str, unit))}")
