"""The arms. Every one is inference only; nothing here trains anything.

Arm A is the configuration Convai's card attributes the published 0.425 to: the
256-token head budget, which is the multilingual and typed-decisions default, not
the English checkpoint's 192. Arm E runs the English checkpoint at its own
defaults so the checkpoint choice is measured separately from the budget.

Arm A doubles as the 256 rung of the budget sweep and the 77-option rung of the
option-count sweep, so it is run once and read three ways.
"""
SEED = 20260924

ARMS = {
    # the published point
    "A-default": {
        "checkpoint": "multilingual", "head_max_len": 256, "max_len": 1024,
        "mode": "flat", "n_options": 77,
    },
    # B: does room alone recover accuracy? Only the budget moves.
    "B-384": {
        "checkpoint": "multilingual", "head_max_len": 384, "max_len": 1024,
        "mode": "flat", "n_options": 77,
    },
    "B-512": {
        "checkpoint": "multilingual", "head_max_len": 512, "max_len": 1024,
        "mode": "flat", "n_options": 77,
    },
    "B-768": {
        "checkpoint": "multilingual", "head_max_len": 768, "max_len": 2048,
        "mode": "flat", "n_options": 77,
    },
    # K: tokens per option move, the configuration does not.
    "K-10": {
        "checkpoint": "multilingual", "head_max_len": 256, "max_len": 1024,
        "mode": "flat", "n_options": 10,
    },
    "K-20": {
        "checkpoint": "multilingual", "head_max_len": 256, "max_len": 1024,
        "mode": "flat", "n_options": 20,
    },
    "K-40": {
        "checkpoint": "multilingual", "head_max_len": 256, "max_len": 1024,
        "mode": "flat", "n_options": 40,
    },
    # C: the workaround the model card documents, at the default budget.
    "C-coarse-fine": {
        "checkpoint": "multilingual", "head_max_len": 256, "max_len": 1024,
        "mode": "coarse-fine", "n_options": 77,
    },
    # E: the English checkpoint at its own defaults.
    "E-english": {
        "checkpoint": None, "head_max_len": 192, "max_len": 512,
        "mode": "flat", "n_options": 77,
    },
}

BUDGET_SWEEP = ["A-default", "B-384", "B-512", "B-768"]
OPTION_SWEEP = ["K-10", "K-20", "K-40", "A-default"]


def resolve(names):
    if names in (None, "", "all"):
        return list(ARMS)
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s): {', '.join(unknown)}; known: {', '.join(ARMS)}")
    return wanted
