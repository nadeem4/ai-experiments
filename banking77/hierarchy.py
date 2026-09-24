"""Arm C: the two-step workaround the model card documents -- choose one of
eleven coarse groups, then the intent inside the chosen group.

Both steps run at the default 256-token budget, so neither step is short of room:
eleven groups and at most eleven labels in a group both fit comfortably.

The reported prediction is the fine step's answer. The joint distribution is
recorded too, because calibration needs a distribution over all 77 labels, but it
is not what the arm predicts: taking the joint argmax would quietly undo the
commitment the coarse step made, and then the arm would no longer be the
workaround anyone would actually deploy.
"""


def coarse_labels(groups):
    return list(groups)


def fine_labels(groups, group):
    return list(groups[group])


def prediction(chosen_group, fine_probs):
    return max(fine_probs, key=lambda label: fine_probs[label])


def joint_probabilities(groups, coarse_probs, chosen_group, fine_probs):
    """P(label) = P(group) * P(label | group). Only the chosen group was scored at
    the fine step, so every other group spreads its own mass evenly over the
    labels it holds -- the only thing the run actually measured about them."""
    joint = {}
    for group, labels in groups.items():
        mass = coarse_probs.get(group, 0.0)
        if group == chosen_group:
            for label in labels:
                joint[label] = mass * fine_probs.get(label, 0.0)
        else:
            for label in labels:
                joint[label] = mass / len(labels)
    return joint
