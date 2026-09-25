"""Which models this experiment runs, as configuration rather than code.

Adding a model must not mean editing the runner. A **hosted** model is one line
here mapping a CLI name to an OpenRouter catalogue prefix -- the exact id is
still resolved live, current-generation-first, and probed for entitlement, so
nothing is hardcoded. A **local** model is one line naming an adapter
`module:Class`; adding a second local decision model later is one new module and
one line, and neither `run.py` nor `report.py` learns its name.

## Who is on the slate, and who is not

Seven hosted families spanning the open and closed ends of the market, plus the
two decision models.

**Gemini is not here.** In an earlier run of this harness it emitted 318
completion tokens to answer a one-word multiple-choice question and would have
been roughly 80% of the bill on its own.

**Claude is not here.** The only current-generation Claude this account can reach
is an Opus tier. Nobody deploys a flagship reasoning model to classify support
intents, so putting one in a cost-per-decision table produces an answer to a
question no one asked.

**Qwen is deliberately not the flagship.** `-max` and `-prime` tiers are denied
in `catalog.py`, so the current-generation pick is the cheap tier, the same rule
every other family gets.

## The probability column

`returns_probability` is a capability, recorded as a first-class column rather
than as a footnote. Jev and Laya return a distribution over every option; an LLM
in structured mode returns a label and nothing else. Without a probability you
cannot set a threshold, so "ask a human when unsure" -- the thing most production
classifiers actually need -- is not available from that model at any price. It is
also why calibration is reported for some rows and is simply absent for others.
"""
import importlib

# CLI name -> OpenRouter catalogue prefix. The id is resolved live from the
# prefix; only the family is written down. `--models name=exact/id` overrides.
HOSTED = {
    "glm": "z-ai/glm-",
    "phi": "microsoft/phi",
    "gemma": "google/gemma-",
    "deepseek": "deepseek/deepseek-",
    "llama": "meta-llama/llama-",
    "qwen": "qwen/qwen",
    "gpt": "openai/gpt-",
}

# Hosted, but on the decision route and with a pinned build. `typesafe/jev-1.13`
# is a floating minor and `~typesafe/jev-latest` is an alias; neither is safe to
# benchmark against, so the dated build is named here.
DECISION = {
    "jev": {"model_id": "typesafe/jev-1.13-20260917", "route": "POST /api/v1/systemone"},
}

# Local weights. One line, one adapter module.
LOCAL = {
    "laya": {"adapter": "typed_decisions.laya_local:LayaModel"},
}

ALL = tuple(list(HOSTED) + list(DECISION) + list(LOCAL))

# Models that hand back a distribution over the options rather than a label.
_PROBABILISTIC = frozenset(list(DECISION) + list(LOCAL))


def kind_of(name):
    if name in HOSTED:
        return "hosted"
    if name in DECISION:
        return "decision"
    if name in LOCAL:
        return "local"
    raise SystemExit(f"unknown model {name!r}; known: {', '.join(ALL)}")


def is_local(name):
    """Local CPU against a hosted API is not a latency comparison, so the
    grouping in every table is mechanical rather than remembered."""
    return kind_of(name) == "local"


def returns_probability(name):
    return name in _PROBABILISTIC


def parse(argument):
    """`all`, or `glm,phi,jev`, or `glm=z-ai/glm-5.3-flash` to pin one exactly.

    -> {name: pinned id or None}, in registry order for `all` so a run's model
    order does not depend on how the flag was typed."""
    if argument.strip() == "all":
        return {name: None for name in ALL}
    out = {}
    for item in argument.split(","):
        name, _, pinned = item.strip().partition("=")
        if name not in ALL:
            raise SystemExit(f"unknown model {name!r}; known: {', '.join(ALL)}")
        out[name] = pinned.strip() or None
    return out


def load_adapter(path):
    """`package.module:Class` -> the class. The whole extension point for a local
    model: write the adapter, name it in LOCAL, done."""
    module_name, _, attribute = path.partition(":")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"cannot load the adapter {path!r}: {type(e).__name__}: {e}")
