"""`cli dataset <name>` -> `<name>/DATASET.md`, written from the real files.

Each experiment reports what it loads; the rendering is here so every
`DATASET.md` in the repository has the same shape and a reader learns it once.

Generated rather than written because a dataset description is a fact that has
to be remembered, and a remembered fact is one that can be confidently wrong
about a split it no longer uses.
"""
SAMPLE_CLASSES = 12
SAMPLE_ROWS = 3


def _table(pairs):
    lines = ["| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in pairs if v is not None]
    return "\n".join(lines)


def _classes(names):
    """Names, or a count and a sample when there are too many to read."""
    if len(names) <= SAMPLE_CLASSES:
        return f"{len(names)}: " + ", ".join(f"`{n}`" for n in names)
    shown = ", ".join(f"`{n}`" for n in names[:SAMPLE_CLASSES])
    return f"**{len(names)}**, including {shown} ..."


def _rows(rows):
    out = []
    for row in rows[:SAMPLE_ROWS]:
        out.append("```")
        for key, value in row.items():
            text = str(value).replace("\n", " ")
            out.append(f"{key}: {text[:300]}{'...' if len(text) > 300 else ''}")
        out.append("```")
    return "\n".join(out)


def section(profile):
    splits = ", ".join(f"**{name}** {count:,}" for name, count in profile["splits"].items())
    pairs = [
        ("Source", f"[{profile['source']}]({profile['url']})"),
        ("Licence", profile["licence"] or "not recorded by this repository"),
        ("Splits", splits),
        ("Used here", profile["used"]),
    ]
    if profile.get("classes"):
        pairs.append(("Labels", _classes(profile["classes"])))
    for key, value in (profile.get("extra") or {}).items():
        pairs.append((key, value))

    return (f"## {profile['name']}\n\n{profile['what_it_is']}\n\n"
            f"{_table(pairs)}\n\n**What a row looks like**\n\n{_rows(profile['rows'])}\n")


def render(profiles):
    head = ("# The data\n\n"
            "Written by `uv run cli dataset <name>` from the files the experiment "
            "actually loads. Do not edit it by hand.\n\n")
    return head + "\n".join(section(p) for p in profiles)


def write(name, profiles):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / name / "DATASET.md"
    path.write_text(render(profiles), encoding="utf-8")
    return path
