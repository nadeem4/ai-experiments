# The data

Written by `uv run cli dataset <name>` from the files the experiment actually loads. Do not edit it by hand.

## BEIR NFCorpus

Health questions from NutritionFacts.org over PubMed abstracts, with human relevance judgements. A query is a question a person asked; a document is a paper that may or may not answer it.

| | |
|---|---|
| Source | [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) |
| Licence | not recorded by this repository |
| Splits | **documents** 3,633, **test queries** 323 |
| Used here | `test`, all 323 queries |
| Judgements | 12,334 query-document pairs marked relevant, a median of 16 per query |
| Grades | 1 or 2, never 0: 11,758 marked relevant and 576 highly so. Anything above 0 counts as relevant here |

**What a row looks like**

```
query_id: PLAIN-102
query: Stopping Heart Disease in Childhood
```
```
a document judged relevant to it: MED-3253
title: Pathobiological determinants of atherosclerosis in youth risk scores are associated with early and advanced atherosclerosis.
text: OBJECTIVES: Atherosclerosis begins in childhood and progresses during adolescence and young adulthood. The Pathobiological Determinants of Atherosclerosis in Youth Study previously reported risk scores to estimate the probability of advanced atherosclerotic lesions in young individuals aged 15 to 34...
```
