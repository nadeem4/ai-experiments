"""Which model each family is represented by, decided by OpenRouter, not by me.

Model ids are not written down anywhere in this experiment. OpenRouter is asked
what it hosts, the catalog is filtered and ranked, and then -- because the catalog
lists far more models than an account is entitled to call -- the ranked candidates
are probed with one tiny request until one answers. The id that answers, and the
version OpenRouter reports for it, go into `config.json`.

The ranking is **current generation first, cheapest tier within it**. The earlier
Vercel-gateway version of this file ranked on price alone, which on OpenRouter
resolved every family to a 2025 model even where the current one was reachable --
the silent substitution the protocol forbids. "Cheapest tier" was always the
intent (the nano / lite / flash class), and it still is; it is now applied inside
the current generation rather than across releases. A family whose current model
is *refused* still falls back down the ladder, and every refusal is recorded so
the report can say which generation actually carried the family.

Two OpenRouter facts this file exists to encode:

  * The decision models are **absent from the default `/api/v1/models` listing**
    and appear only under `?output_modalities=decisions`. So they are fetched
    separately, and `JEV_ID` is a pinned dated version rather than something
    ranked -- `typesafe/jev-1.13` and `~typesafe/jev-latest` both float.
  * Pricing is spelled `prompt` / `completion` here, where the Vercel gateway
    spelled it `input` / `output`. Reading the wrong key prices everything at zero
    silently, so it is read in exactly one place.
"""
import datetime
import json
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
DECISIONS_URL = f"{MODELS_URL}?output_modalities=decisions"

# Which families this experiment runs is `registry.HOSTED`, not this file; the
# alias is kept so anything still importing `catalog.FAMILIES` gets the slate.
from .registry import HOSTED as FAMILIES  # noqa: E402  (a re-export, not a cycle)

# Variants that are the same family at a different price or a different job. Left
# in, the "cheapest" pick would be arbitrary -- a `-fast` mirror, a reasoning tier,
# an image model -- rather than the family's ordinary decision-making model.
# `batch` is here because `:batch` is an asynchronous queue at half price: a
# different product with a different latency, which must never be averaged into a
# latency table, and `:free` is the same objection at a rate-limited mirror.
# `-max` and `prime` are the flagship tiers: left in, Qwen's current generation
# resolved to a $2/M `-max` and a $4/M `-prime`, which is the opposite of the
# cheapest-tier rule every other family gets.
DENY = ("-fast", "-pro", "codex", "thinking", "image", "safeguard", "realtime",
        "oss", "preview", "omni", "batch", "search", "audio", "tts",
        "-max", "prime", ":free")

# Pinned, not ranked. `typesafe/jev-1.13` is the floating minor and
# `~typesafe/jev-latest` is the alias; neither is safe to benchmark against.
JEV_ID = "typesafe/jev-1.13-20260917"


def fetch(api_key, url=MODELS_URL):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.load(r)["data"]


def fetch_decisions(api_key):
    """The decision models, which the default listing does not carry at all."""
    return fetch(api_key, DECISIONS_URL)


def denied(model_id):
    return any(d in model_id for d in DENY)


def price(model):
    """OpenRouter's prompt price, per token, as a float."""
    return float((model.get("pricing") or {}).get("prompt") or 0.0)


def _month(created):
    if not created:
        return None
    return datetime.datetime.fromtimestamp(created, datetime.timezone.utc).strftime("%Y-%m")


def generation(model):
    """A release-month bucket. Two models shipped the same month are the same
    generation and are then ranked on price; a month apart is a different rung."""
    created = model.get("created")
    if not created:
        return 0
    d = datetime.datetime.fromtimestamp(created, datetime.timezone.utc)
    return d.year * 12 + d.month


def _supports(model, parameter):
    return parameter in (model.get("supported_parameters") or [])


def accepts_temperature(model):
    """The current GPT and Claude tiers do not. Fairness rule 3: temperature 0
    where the model takes one, null where it does not, recorded either way --
    never silently assumed, and never a reason to exclude the model."""
    return _supports(model, "temperature")


def accepts_reasoning_effort(model):
    """The current tiers are reasoning models, and their thinking tokens can eat a
    small answer budget whole. Where the model advertises the flag it is asked for
    a decision rather than an essay; where it does not, nothing is sent."""
    return _supports(model, "reasoning_effort")


def is_text_model(model):
    """`["text"]` is an LLM; `["decisions"]` is Jev, which answers on another route
    entirely and must never be picked as though it were a chat model."""
    return (model.get("architecture") or {}).get("output_modalities") == ["text"]


def candidates(models, prefix):
    """Every model in the family that could carry this experiment, current
    generation first and cheapest tier within it.

    `structured_outputs` is required because rule one of the fairness rules is to
    give the LLMs their best shot: a model that cannot be constrained to the option
    list is being asked a different, easier-to-fail question. A temperature is
    **not** required -- requiring one is what silently excluded every
    current-generation model -- and is recorded per model instead."""
    out = [
        m for m in models
        if m["id"].startswith(prefix)
        and is_text_model(m)
        and _supports(m, "structured_outputs")
        and not denied(m["id"])
    ]
    out.sort(key=lambda m: (-generation(m), price(m), m["id"]))
    return out


def resolve(models, prefix, probe):
    """-> (model or None, [(id, reachable), ...]).

    The catalog is what OpenRouter hosts; `probe` is what this account may call.
    The whole ladder is recorded, so a report can say which models were refused and
    at which generation the family actually ran."""
    tried = []
    for model in candidates(models, prefix):
        reachable = bool(probe(model["id"]))
        tried.append((model["id"], reachable))
        if reachable:
            return model, tried
    return None, tried


def version(model):
    """What OpenRouter says about the model, kept verbatim in `config.json`.

    `canonical_slug` is the dated build behind a floating id, and it is the one a
    benchmark records: the id can move under you, the slug cannot."""
    return {
        "id": model["id"],
        "canonical_slug": model.get("canonical_slug"),
        "name": model.get("name"),
        "created": model.get("created"),
        "released": _month(model.get("created")),
        "context_length": model.get("context_length"),
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "pricing": model.get("pricing"),
        "supported_parameters": model.get("supported_parameters"),
    }


def find(models, model_id):
    return next((m for m in models if m["id"] == model_id), None)
