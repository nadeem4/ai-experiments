"""Is the number a model hands back a probability you could act on?

Only the decision models answer that question at all: Jev and Laya return a
distribution over every option, and an LLM in structured mode returns a label and
nothing else. That is not a small difference. Without a probability there is no
threshold, so "route this one to a human" is not available from that model at any
price -- which is why the capability is a column in the report and calibration is
present for some rows and simply absent for others rather than shown as zero.

Two numbers per probabilistic model:

  * **ECE over 15 bins, raw.** The gap between how confident the model was and
    how often it was right, averaged over confidence bins and weighted by how
    many answers fell in each. A bin with nothing in it contributes nothing; an
    empty bin is not evidence of calibration.
  * **ECE after one temperature.** A single scalar `T` dividing the log
    probabilities before re-normalising. It cannot change the argmax, so it
    cannot change accuracy: it is a calibration fix and reporting it as anything
    else would be wrong.

**The temperature is fitted on a validation split carved out of train, never on
test.** `calibrate()` takes the two sets separately for exactly that reason -- the
fit never sees a test row, and the test above proves a different test set does
not move the fitted value.
"""
import math

N_BINS = 15
_GRID = [0.02 * i for i in range(1, 501)]  # T in (0.02, 10.0], refined below


def ece(confidences, correct, n_bins=N_BINS):
    """Expected calibration error: sum over bins of |accuracy - confidence|,
    weighted by the share of answers in the bin."""
    if len(confidences) != len(correct):
        raise ValueError(f"{len(confidences)} confidences against {len(correct)} outcomes")
    if not confidences:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    total = len(confidences)
    return sum(
        len(rows) / total * abs(sum(ok for _, ok in rows) / len(rows)
                                - sum(c for c, _ in rows) / len(rows))
        for rows in bins if rows
    )


def _log(p):
    """log with a floor, because a model that returns a hard 0 for an option is
    common and -inf would end the fit rather than inform it."""
    return math.log(max(float(p), 1e-12))


def _nll(rows, temperature):
    total = 0.0
    for probs, gold in rows:
        scaled = {k: _log(v) / temperature for k, v in probs.items()}
        peak = max(scaled.values())
        denominator = sum(math.exp(v - peak) for v in scaled.values())
        total -= (scaled.get(gold, _log(0) / temperature) - peak) - math.log(denominator)
    return total / len(rows)


def fit_temperature(rows):
    """-> the scalar T minimising negative log likelihood on `rows`.

    `rows` is [({option: probability}, gold option)] and must come from the
    **validation** split. A coarse grid then a local refinement: the objective is
    one-dimensional and smooth, so this is exact enough and adds no dependency.

    `None` when there is nothing to fit on. Returning 1.0 instead would report an
    unfitted model as though it had been calibrated."""
    if not rows:
        return None
    best = min(_GRID, key=lambda t: _nll(rows, t))
    fine = [best + 0.002 * i for i in range(-9, 10) if best + 0.002 * i > 0]
    return min(fine, key=lambda t: _nll(rows, t))


def apply_temperature(probs, temperature):
    """Softmax of (log p / T). T = 1 is the identity; T > 1 flattens; T < 1
    sharpens. The ranking is untouched at every T, so accuracy cannot move."""
    scaled = {k: _log(v) / temperature for k, v in probs.items()}
    peak = max(scaled.values())
    exponentiated = {k: math.exp(v - peak) for k, v in scaled.items()}
    total = sum(exponentiated.values())
    return {k: v / total for k, v in exponentiated.items()}


def calibrate(validation, test, n_bins=N_BINS):
    """-> {"temperature", "ece_raw", "ece_scaled", "n_validation", "n_test"}.

    `None` when the model returned no probabilities. A model that cannot be
    calibrated is absent from the calibration table rather than scored zero,
    which would read as perfectly calibrated."""
    if not test:
        return None
    temperature = fit_temperature(validation)
    raw_conf = [max(p.values()) for p, _ in test]
    raw_correct = [int(max(p, key=p.get) == gold) for p, gold in test]
    out = {
        "temperature": temperature,
        "ece_raw": ece(raw_conf, raw_correct, n_bins),
        "ece_scaled": None,
        "n_validation": len(validation),
        "n_test": len(test),
        "n_bins": n_bins,
    }
    if temperature is not None:
        scaled = [apply_temperature(p, temperature) for p, _ in test]
        out["ece_scaled"] = ece([max(p.values()) for p in scaled], raw_correct, n_bins)
    return out


def reliability(confidences, correct, n_bins=N_BINS):
    """-> {"bin_centres", "accuracy", "counts"} for the reliability diagram.

    One point per **non-empty** bin: a bin nothing fell into is not a model that
    was wrong there, and plotting it at zero would draw a curve the run never
    measured. The x of each point is the mean confidence actually expressed in
    that bin rather than the nominal centre of the bucket, because the curve is
    read against the diagonal."""
    if len(confidences) != len(correct):
        raise ValueError(f"{len(confidences)} confidences against {len(correct)} outcomes")
    if not confidences:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    filled = [rows for rows in bins if rows]
    return {
        "bin_centres": [sum(c for c, _ in rows) / len(rows) for rows in filled],
        "accuracy": [sum(ok for _, ok in rows) / len(rows) for rows in filled],
        "counts": [len(rows) for rows in filled],
    }
