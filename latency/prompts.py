"""The same question, rendered for a transport that only speaks text.

A decision model is handed `instructions` and a `criteria` map and answers by
scoring each option at its own marker. An LLM has to be handed the same thing as a
prompt. Everything here is mechanical on purpose: the instruction line goes in
verbatim, every option key and description goes in verbatim and in the given order,
and the only thing added is `ANSWER_DIRECTIVE`, which is one constant shared by
every LLM. That is the whole difference between the arms -- the transport -- and it
is kept in one file so it can be read in thirty seconds.
"""

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
