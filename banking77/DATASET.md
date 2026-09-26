# The data

Written by `uv run cli dataset <name>` from the files the experiment actually loads. Do not edit it by hand.

## BANKING77

Real customer queries to a banking assistant, each labelled with one of 77 fine-grained intents. The intents are close together on purpose, which is what makes it hard.

| | |
|---|---|
| Source | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) |
| Licence | CC-BY-4.0 |
| Splits | **train** 10,003, **test** 3,080 |
| Used here | `test`, all 3,080 rows |
| Labels | **77**, including `Refund_not_showing_up`, `activate_my_card`, `age_limit`, `apple_pay_or_google_pay`, `atm_support`, `automatic_top_up`, `balance_not_updated_after_bank_transfer`, `balance_not_updated_after_cheque_or_cash_deposit`, `beneficiary_not_allowed`, `cancel_transfer`, `card_about_to_expire`, `card_acceptance` ... |
| Balance | 40 test rows per intent, exactly |

**What a row looks like**

```
text: How do I locate my card?
label: card_arrival
```
```
text: I still have not received my new card, I ordered over a week ago.
label: card_arrival
```
```
text: I ordered a card but it has not arrived. Help please!
label: card_arrival
```
