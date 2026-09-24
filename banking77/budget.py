"""What the token budget actually did to the options, read off the real request.

laya builds `[CLS] head [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]` and
returns the marker position of every option. Nothing here re-implements that: the
spans are cut out of the sequence laya itself built, so the diagnostics describe
the bytes the model saw rather than a model of them.

The count that matters is `identical_pairs`. Two labels that truncate to the same
string cannot be told apart by any amount of capability, so that number is the
budget hypothesis stated as evidence rather than as a claim.
"""


def option_spans(ids, markers, sep_id):
    """-> one token-id list per option, `[MASK]` marker included. The last option
    runs to the separator that closes the option block."""
    if any(m >= len(ids) for m in markers):
        raise ValueError(f"marker past the end of a {len(ids)}-token sequence: {markers}")
    spans = []
    for i, start in enumerate(markers):
        if i + 1 < len(markers):
            end = markers[i + 1]
        else:
            end = ids.index(sep_id, start)
        spans.append(list(ids[start:end]))
    return spans


def truncation_stats(spans, full_lengths, decoded):
    """`full_lengths` is what each option would have been without the budget."""
    if not (len(spans) == len(full_lengths) == len(decoded)):
        raise ValueError("spans, full_lengths and decoded must describe the same options")
    n = len(spans)
    seen = {}
    for text in decoded:
        seen[text] = seen.get(text, 0) + 1
    return {
        "n_options": n,
        # the marker token is not option text, so it does not count as a token the
        # option got to describe itself with
        "mean_text_tokens_per_option": sum(len(s) - 1 for s in spans) / n if n else 0.0,
        "min_text_tokens_per_option": min((len(s) - 1 for s in spans), default=0),
        "truncated_fraction": sum(1 for s, f in zip(spans, full_lengths) if len(s) < f) / n if n else 0.0,
        "truncated_options": sum(1 for s, f in zip(spans, full_lengths) if len(s) < f),
        "distinct_options": len(seen),
        "identical_pairs": sum(c * (c - 1) // 2 for c in seen.values() if c > 1),
        "collided_options": sum(c for c in seen.values() if c > 1),
    }


def laya_tokens_per_option(head_max_len, n_options):
    """laya.common.build_sequence's own arithmetic, for the projected budget before
    a run: when the options do not fit, each is cut to this many tokens, the
    `[MASK]` marker included. Options that already fit are left alone."""
    return max(4, (head_max_len - 16) // max(1, n_options))
