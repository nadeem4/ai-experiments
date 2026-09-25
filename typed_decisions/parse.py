"""Did the call return one of the allowed options, without post-hoc repair?

Validity is a first-class result here, not an error to hide: a model that is fast
and returns prose you cannot parse has not done the job. So the rule is strict.
The answer is the `choice` field of a JSON object, spelled exactly as the option is
spelled. Nothing is fuzzy-matched and nothing is retried -- a repaired answer is a
different product with a different latency, and repairing quietly is how benchmarks
lie. Leading and trailing whitespace is stripped, because that is the transport
rather than the answer.
"""
import json

KINDS = ("valid", "unparseable", "not_an_option", "refusal", "api_error")

# A refusal and a broken completion are both failures, but they are different
# failures, so prose is checked against a short fixed list of refusal openers
# before being written off as unparseable. The list is deliberately small: a model
# that refuses says so in the first clause.
REFUSAL_MARKERS = ("i'm sorry", "i am sorry", "i can't", "i cannot", "i won't", "i'm unable", "as an ai")


def _out(choice, validity, detail=None):
    return {"choice": choice, "validity": validity, "detail": detail}


def classify(content, options):
    if content is None:
        return _out(None, "api_error", "no content")
    text = content.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        lowered = text.lower()
        if any(lowered.startswith(m) or m in lowered[:80] for m in REFUSAL_MARKERS):
            return _out(None, "refusal", text[:200])
        return _out(None, "unparseable", text[:200] or "empty")
    if not isinstance(parsed, dict):
        return _out(None, "unparseable", "not an object")
    if "choice" not in parsed:
        return _out(None, "unparseable", "no choice field")
    choice = parsed["choice"]
    if choice in options:
        return _out(choice, "valid")
    return _out(None, "not_an_option", str(choice)[:200])
