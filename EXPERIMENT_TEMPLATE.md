# Experiment template

Every experiment in this repo is two files:

```
<experiment>/
  README.md     the protocol   written before the run
  RESULTS.md    what happened  written after
```

Two files rather than two sections of one file, because the parts written before a run
and the parts written after must be separable. A prediction that shares a file with the
answer can be quietly reworded once the answer is known. Split, `git log` on each file
shows which one came first.

**Protocol amendments are appended to the bottom of README.md with a date. A section is
never edited in place once the run has started.**

Copy the two skeletons below. Each note says what belongs in the section and why. If a
section has no material, write one honest line saying so rather than padding it.

---

## README.md

````markdown
# <the experiment, written as its question>

Results: [RESULTS.md](RESULTS.md), once there are some. How far this got is
read off `results/` by `uv run cli list`, not written down here.
<!-- designed: nothing has run. piloted: a small run has. complete: the full run has. -->

## The question
<!-- One sentence, answerable with a yes/no or a number. Two sentences means two experiments. -->

## Motivation
<!-- Why it is worth doing, and what changes depending on the answer. If nothing changes, do not run it. -->

## What we expect
<!-- The prediction, and what would falsify it. Written before the run, or labelled in plain
     words as reconstructed after it. A reconstruction is not evidence and must not read like one. -->

## Data
<!-- Source, version, exact split, licence, how to obtain it. Enough for someone else to fetch the same bytes. -->

## Method
<!-- The arms or conditions, what varies between them, what is held fixed, every seed. -->

## Metrics
<!-- Each one named, with why it is there. A metric with no reason is a metric nobody reads. -->

## Assumptions and limits
<!-- Hardware, sample size, anything not comparable across arms. Caveats live here, not in a
     footnote under the results, because they bound the result before it exists. -->

## How to run it
<!-- The actual commands, the expected wall clock and the expected cost. -->

## Protocol amendments
<!-- Appended, dated, never edited in place. Omit the section until there is one. -->
````

---

## RESULTS.md

````markdown
# Results: <the experiment>

## The finding
<!-- One sentence, at the top, before any table. -->

## The numbers
<!-- The table, with intervals. Every figure comes from a results file or a run artifact. -->

## Did the prediction hold?
<!-- Explicitly, including when it did not. A wrong prediction is the most useful line in the file. -->

## What surprised us
<!-- What the protocol did not anticipate. "Nothing" is a legitimate answer. -->

## What this does not answer
<!-- The questions a reader will ask next that this run cannot settle. -->

## Provenance
<!-- Model ids and versions, library versions, date, commit, hardware. -->
````
