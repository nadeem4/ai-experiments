"""The option set, frozen once so no arm is measuring option order.

Options are the 77 BANKING77 intent labels with underscores replaced by spaces
and nothing else -- no hand-written descriptions. Descriptions would make every
option longer and so make the truncation worse, and the point of the experiment
is to find out whether the label text is the binding constraint at all.

`COARSE_GROUPS` is arm C's two-step grouping. It is written here, before any
model has run, and frozen to `option_texts/` alongside the flat list.
"""
import json
from pathlib import Path

# The 77 intents, in the order every arm renders them: sorted, which is also the
# order the frozen file records. `Refund_not_showing_up` is capitalised upstream
# and so sorts first; that quirk is upstream's and is kept rather than fixed.
ALL_LABELS = sorted([
    "Refund_not_showing_up", "activate_my_card", "age_limit", "apple_pay_or_google_pay",
    "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
])

# Eleven groups, written by hand before the run. Every label appears exactly once.
COARSE_GROUPS = {
    "getting a card": [
        "card_arrival", "card_delivery_estimate", "order_physical_card", "get_physical_card",
        "getting_spare_card", "card_linking", "activate_my_card",
    ],
    "virtual and disposable cards": [
        "get_disposable_virtual_card", "getting_virtual_card", "disposable_card_limits",
        "virtual_card_not_working",
    ],
    "a card that is broken, lost or locked": [
        "card_not_working", "contactless_not_working", "card_swallowed", "lost_or_stolen_card",
        "lost_or_stolen_phone", "compromised_card", "card_about_to_expire", "pin_blocked",
        "change_pin", "passcode_forgotten",
    ],
    "card payments and charges on the statement": [
        "declined_card_payment", "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "pending_card_payment", "reverted_card_payment?",
        "direct_debit_payment_not_recognised", "extra_charge_on_statement",
        "transaction_charged_twice",
    ],
    "cash machines and withdrawals": [
        "atm_support", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal", "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "transfers and money coming in": [
        "cancel_transfer", "declined_transfer", "failed_transfer", "pending_transfer",
        "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
        "transfer_timing", "beneficiary_not_allowed", "balance_not_updated_after_bank_transfer",
        "receiving_money",
    ],
    "topping up the account": [
        "automatic_top_up", "pending_top_up", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
        "top_up_reverted", "topping_up_by_card", "verify_top_up",
        "balance_not_updated_after_cheque_or_cash_deposit",
    ],
    "refunds": ["Refund_not_showing_up", "request_refund"],
    "currencies, exchange rates and where the card works": [
        "exchange_charge", "exchange_rate", "exchange_via_app", "fiat_currency_support",
        "supported_cards_and_currencies", "visa_or_mastercard", "card_acceptance",
        "country_support",
    ],
    "proving who you are": [
        "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "why_verify_identity", "age_limit",
    ],
    "the account and the app": [
        "edit_personal_details", "terminate_account", "apple_pay_or_google_pay",
    ],
}


def option_text(label):
    return label.replace("_", " ")


def option_texts(labels):
    return [option_text(label) for label in labels]


def criteria(texts):
    """laya renders a choice option as `key` when its value is None, so the key is
    the whole option text and no description is added."""
    return {text: None for text in texts}


def group_of(label):
    for group, labels in COARSE_GROUPS.items():
        if label in labels:
            return group
    raise KeyError(label)


def subset_labels(n, seed):
    """Arm K's label subsets: a seeded sample, nested so that the 10-label set sits
    inside the 20-label set, and always returned in the frozen option order."""
    import random

    if n >= len(ALL_LABELS):
        return list(ALL_LABELS)
    shuffled = list(ALL_LABELS)
    random.Random(seed).shuffle(shuffled)
    picked = set(shuffled[:n])
    return [label for label in ALL_LABELS if label in picked]


def freeze(path, labels):
    """Writes the option set once. A second call with a different order raises
    rather than quietly re-ordering the options under a half-finished run."""
    path = Path(path)
    payload = {
        "labels": list(labels),
        "option_texts": option_texts(labels),
        "coarse_groups": COARSE_GROUPS,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["labels"] != list(labels):
            raise ValueError(f"{path} is already frozen with a different option order")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["labels"]
