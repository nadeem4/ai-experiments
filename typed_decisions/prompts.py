"""The text transport: the question on the way out, the answer on the way back.

An LLM cannot be handed a typed question, so it is handed a prompt, and its
completion has to be read back into a decision. Both ends live here because they
are one interface, and because this is the one place the experiment could put a
thumb on the scale -- so both are mechanical and short enough to read in a
minute.

A decision model is handed `instructions` and a `criteria` map and answers by
scoring each option at its own marker. An LLM has to be handed the same thing as a
prompt. Everything here is mechanical on purpose: the instruction line goes in
verbatim, every option key and description goes in verbatim and in the given order,
and the only thing added is `ANSWER_DIRECTIVE`, which is one constant shared by
every LLM. That is the whole difference between the arms -- the transport -- and it
is kept in one file so it can be read in thirty seconds.
"""

import json

ANSWER_DIRECTIVE = 'Answer with a JSON object {"choice": "<one option, copied exactly>"} and nothing else.'


def option_block(criteria):
    """`KEY: description` per line, or the bare key where an option has no
    description -- which is how laya renders a criterion whose value is None."""
    lines = [key if value is None else f"{key}: {value}" for key, value in criteria.items()]
    return "\n".join(lines)


def state_block(state):
    """A decision model takes the state as a dict of fields or as a string. Both go
    in as they are; nothing is summarised or reordered."""
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return str(state)


def render(instructions, state, criteria, with_options=True):
    """The user message. `with_options=False` is the same prompt with the option
    list removed and nothing else changed, so differencing the provider's reported
    prompt tokens across the two measures what the options alone cost."""
    parts = [instructions, "", "Situation:", state_block(state), ""]
    if with_options:
        parts += ["Options:", option_block(criteria), ""]
    parts.append(ANSWER_DIRECTIVE)
    return "\n".join(parts)


def schema(criteria, with_enum=True):
    """Structured output pinned to the option keys: the strongest constraint the
    gateway exposes, so the LLM is given its best shot at a valid answer.

    `with_enum=False` is the same schema with the option list taken out of it, and
    exists only for the overhead probe. Some providers bill a json_schema as a tool
    definition, so the probe's second call has to keep the scaffolding and drop only
    the options -- otherwise the scaffolding is charged to the options."""
    choice = {"type": "string", "enum": list(criteria)} if with_enum else {"type": "string"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"choice": choice},
                "required": ["choice"],
                "additionalProperties": False,
            },
        },
    }


# --- reading the answer back --------------------------------------------------

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
