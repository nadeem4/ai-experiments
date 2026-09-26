# The data

Written by `uv run cli dataset <name>` from the files the experiment actually loads. Do not edit it by hand.

## ag_news (fancyzhx/ag_news)

News stories, each belonging to one of four sections. The easy end of the option-count axis: four labels a person could hold in mind.

| | |
|---|---|
| Source | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) |
| Licence | unknown -- the hub lists `license: unknown` for this dataset. Recorded as unknown rather than assumed permissive. |
| Splits | **test** 7,600 |
| Used here | `test`, config `default`. The run samples 300 examples from it |
| Labels | 4: `World`, `Sports`, `Business`, `Sci/Tech` |

**What a row looks like**

```
text: Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.
label: Business
```
```
text: The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date f...
label: Sci/Tech
```
```
text: Ky. Company Wins Grant to Study Peptides (AP) AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides, which are short chains of amino acids, the building blocks of proteins.
label: Sci/Tech
```

## clinc150 (clinc/clinc_oos)

Utterances to a voice assistant, each one of 150 intents plus out-of-scope. The hard end: 151 options, many a sentence apart.

| | |
|---|---|
| Source | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) |
| Licence | CC-BY-3.0 |
| Splits | **test** 5,500 |
| Used here | `test`, config `plus`. The run samples 300 examples from it |
| Labels | **151**, including `restaurant_reviews`, `nutrition_info`, `account_blocked`, `oil_change_how`, `time`, `weather`, `redeem_rewards`, `interest_rate`, `gas_type`, `accept_reservations`, `smart_home`, `user_name` ... |

**What a row looks like**

```
text: how would you say fly in italian
label: translate
```
```
text: what's the spanish word for pasta
label: translate
```
```
text: how would they say butter in zambia
label: translate
```
