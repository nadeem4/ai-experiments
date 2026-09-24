"""Which model each family is represented by, decided by the gateway, not by me.

Model ids are not written down anywhere in this experiment. The gateway is asked
what it hosts, the catalog is filtered and ranked, and then -- because the catalog
lists far more models than an account is entitled to call -- the ranked candidates
are probed with one tiny request until one answers. The id that answers, and the
version the gateway reports for it, go into `config.json`.

The ranking is cheapest first. This experiment measures what a decision costs, so
each family is represented by the cheapest tier that can still be asked in
structured mode; a family whose cheapest reachable model is old is a fact about
the account, and the report says so rather than substituting a different family.
"""
import json
import urllib.request

MODELS_URL = "https://ai-gateway.vercel.sh/v1/models"

FAMILIES = {
    "gpt": "openai/gpt-",
    "claude": "anthropic/claude-",
    "gemini": "google/gemini-",
    "gemma": "google/gemma-",
}

# Variants that are the same family at a different price or a different job. Left
# in, the "cheapest" pick would be arbitrary -- a `-fast` mirror, a reasoning tier,
# an image model -- rather than the family's ordinary decision-making model.
DENY = ("-fast", "-pro", "codex", "thinking", "image", "safeguard", "realtime", "oss", "preview", "omni")

JEV_ID = "typesafe-ai/jev"


def fetch(api_key, url=MODELS_URL):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.load(r)["data"]


def denied(model_id):
    return any(d in model_id for d in DENY)


def _price(model):
    return float((model.get("pricing") or {}).get("input") or 0.0)


def candidates(models, prefix):
    """Every model in the family that could carry this experiment, cheapest first.

    `structured_outputs` is required because rule one of the fairness rules is to
    give the LLMs their best shot: a model that cannot be constrained to the option
    list is being asked a different, easier-to-fail question. A temperature is
    required too, so that every arm is at zero and the one that cannot be is an
    explicit exception rather than a silent one."""
    out = [
        m for m in models
        if m["id"].startswith(prefix)
        and m.get("type") == "language"
        and "structured_outputs" in (m.get("supported_parameters") or [])
        and m.get("temperature")
        and not denied(m["id"])
    ]
    out.sort(key=lambda m: (_price(m), -(m.get("released") or 0)))
    return out


def resolve(models, prefix, probe):
    """-> (model or None, [(id, reachable), ...]).

    The catalog is what the gateway hosts; `probe` is what this account may call.
    The whole ladder is recorded, so a report can say which models were refused."""
    tried = []
    for model in candidates(models, prefix):
        reachable = bool(probe(model["id"]))
        tried.append((model["id"], reachable))
        if reachable:
            return model, tried
    return None, tried


def version(model):
    """What the gateway says about the model, kept verbatim in `config.json`."""
    return {
        "id": model["id"],
        "name": model.get("name"),
        "released": model.get("released"),
        "context_window": model.get("context_window"),
        "knowledge": model.get("knowledge"),
        "pricing": model.get("pricing"),
    }


def find(models, model_id):
    return next((m for m in models if m["id"] == model_id), None)
