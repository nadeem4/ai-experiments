from banking77 import arms


def test_arm_a_is_the_published_configuration_that_produced_0_425():
    a = arms.ARMS["A-default"]
    assert a["checkpoint"] == "multilingual"
    assert a["head_max_len"] == 256 and a["max_len"] == 1024
    assert a["mode"] == "flat" and a["n_options"] == 77


def test_arm_e_is_the_english_checkpoint_at_its_own_defaults():
    e = arms.ARMS["E-english"]
    assert e["checkpoint"] is None
    assert e["head_max_len"] == 192 and e["max_len"] == 512


def test_the_budget_sweep_raises_max_len_to_fit_every_head_budget():
    # Arm A is the sweep's 256 rung, so it is not run twice.
    sweep = [name for name in arms.ARMS if name.startswith("B-")]
    assert sorted(arms.ARMS[n]["head_max_len"] for n in sweep) == [384, 512, 768]
    assert arms.BUDGET_SWEEP == ["A-default", "B-384", "B-512", "B-768"]
    for name in sweep:
        a = arms.ARMS[name]
        assert a["max_len"] >= a["head_max_len"] + 256, "the state must keep room too"
        assert a["checkpoint"] == "multilingual", "only the budget may differ from arm A"
        assert a["n_options"] == 77


def test_the_option_count_sweep_holds_the_budget_fixed_at_arm_a():
    # Arm A is the sweep's 77-option rung, so it is not run twice.
    sweep = {n: a for n, a in arms.ARMS.items() if n.startswith("K-")}
    assert sorted(a["n_options"] for a in sweep.values()) == [10, 20, 40]
    for a in sweep.values():
        assert (a["head_max_len"], a["max_len"], a["checkpoint"]) == (256, 1024, "multilingual")
        assert a["mode"] == "flat"


def test_arm_c_is_two_step_at_the_default_budget():
    c = arms.ARMS["C-coarse-fine"]
    assert c["mode"] == "coarse-fine"
    assert (c["head_max_len"], c["max_len"], c["checkpoint"]) == (256, 1024, "multilingual")


def test_every_arm_names_a_known_checkpoint_and_a_known_mode():
    for name, a in arms.ARMS.items():
        assert a["checkpoint"] in (None, "multilingual", "typed-decisions")
        assert a["mode"] in ("flat", "coarse-fine")


def test_resolve_expands_all_and_keeps_the_declared_order():
    assert arms.resolve("all") == list(arms.ARMS)
    assert arms.resolve("A-default,E-english") == ["A-default", "E-english"]


def test_resolve_names_an_unknown_arm_rather_than_silently_skipping_it():
    try:
        arms.resolve("A-default,nope")
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("an unknown arm must raise")


def test_the_option_count_sweep_tops_out_at_arm_a():
    assert arms.OPTION_SWEEP == ["K-10", "K-20", "K-40", "A-default"]
