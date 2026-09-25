# Results

**Status:** `designed`. Nothing has been measured under this protocol yet.

The protocol is [README.md](README.md), and it is committed before any measured
call so that `git log` shows which came first.

The earlier `latency/` and `decision_cost/` versions of this experiment measured
different tasks entirely — a 5-option `highway` task with no ground truth, and a
77-option BANKING77 task imported from a sibling experiment. Neither task exists
here any more, and **none of those numbers are comparable to anything this
protocol will produce**: different datasets, a different model slate, and a
different set of arms. They remain in git history at commit `92dea00` rather than
being reprinted here, because a superseded number sitting next to a current one
is how a wrong figure gets quoted.
