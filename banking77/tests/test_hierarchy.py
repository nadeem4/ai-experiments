from banking77 import hierarchy

GROUPS = {"cards": ["card_arrival", "card_linking"], "transfers": ["failed_transfer"]}


def test_coarse_labels_are_the_group_names_in_declared_order():
    assert hierarchy.coarse_labels(GROUPS) == ["cards", "transfers"]


def test_the_chosen_group_narrows_the_fine_step_to_its_own_labels():
    assert hierarchy.fine_labels(GROUPS, "cards") == ["card_arrival", "card_linking"]


def test_an_unknown_group_raises_rather_than_guessing():
    try:
        hierarchy.fine_labels(GROUPS, "nope")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown group must raise")


def test_the_joint_distribution_multiplies_the_two_steps():
    coarse = {"cards": 0.8, "transfers": 0.2}
    fine = {"card_arrival": 0.75, "card_linking": 0.25}
    joint = hierarchy.joint_probabilities(GROUPS, coarse, "cards", fine)
    assert joint["card_arrival"] == 0.8 * 0.75
    assert joint["card_linking"] == 0.8 * 0.25


def test_labels_outside_the_chosen_group_split_their_own_groups_mass_evenly():
    # Those labels were never scored at the fine step, so the only honest thing to
    # report is the group's own probability spread over the labels it holds.
    coarse = {"cards": 0.8, "transfers": 0.2}
    joint = hierarchy.joint_probabilities(
        GROUPS, coarse, "cards", {"card_arrival": 1.0, "card_linking": 0.0}
    )
    assert joint["failed_transfer"] == 0.2
    assert abs(sum(joint.values()) - 1.0) < 1e-9


def test_the_joint_distribution_covers_every_label():
    joint = hierarchy.joint_probabilities(
        GROUPS, {"cards": 0.5, "transfers": 0.5}, "transfers", {"failed_transfer": 1.0}
    )
    assert set(joint) == {"card_arrival", "card_linking", "failed_transfer"}


def test_the_prediction_is_the_fine_answer_not_the_joint_argmax():
    # The two-step workaround commits to a group and then to a label inside it.
    # Reporting the joint argmax instead would quietly un-commit the first step.
    coarse = {"cards": 0.4, "transfers": 0.6}
    assert hierarchy.prediction("transfers", {"failed_transfer": 1.0}) == "failed_transfer"
    assert coarse  # the coarse step only chooses the group
