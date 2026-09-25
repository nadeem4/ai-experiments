import Link from "next/link";
import banking77 from "@/data/banking77.json";
import results from "@/data/results.json";
import examples from "@/data/examples.json";
import typedDecisions from "@/data/typed-decisions.json";
import { pct, points, type Banking, type TypedDecisions } from "@/lib/experiments";
import { byMethod, signed, type Results } from "@/lib/results";

const run = results as Results;
const jev = byMethod(run, "jev-score")!;
const laya = byMethod(run, "laya-score")!;

const banking = banking77 as Banking;
const typed = typedDecisions as unknown as TypedDecisions;

const bankingDefault = banking.arms.find((a) => a.arm === banking.headline.default_arm)!;
const bankingRaised = banking.arms.find((a) => a.arm === banking.headline.raised_arm)!;

const jevRows = typed.models.filter((m) => m.model === "jev");
const jevSmall = jevRows.reduce((a, b) => (a.n_options < b.n_options ? a : b));
const jevWide = jevRows.reduce((a, b) => (a.n_options > b.n_options ? a : b));
const jevRise = typed.latency_rise.find((r) => r.model === "jev")!;
const jevAccuracyWins = typed.tasks.filter((task) => {
  const scored = typed.models.filter((m) => m.task === task.task && m.accuracy !== null);
  return scored.reduce((a, b) => (a.accuracy! > b.accuracy! ? a : b)).model === "jev";
}).length;
const movedByOrder = typed.position_bias.filter((b) => (b.spread ?? 0) > 0.1).length;

export default function Page() {
  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-24 pt-12 md:px-8">
      <h1 className="max-w-[16ch] text-h1 font-semibold leading-[1.05] tracking-tight">AI Experiments</h1>
      <p className="mt-6 max-w-[62ch] text-lead leading-relaxed text-ink-soft">
        One question at a time, measured on a public benchmark, with the exact request and response kept for
        every call. A result that comes out negative stays negative here.
      </p>

      <h2 className="mt-16 border-t border-line pt-8 text-h3 font-semibold tracking-tight">Experiments</h2>
      <ul className="mt-6 grid gap-4">
        <li>
          <Link
            href="/typed-decisions/"
            className="group block border border-line bg-surface p-5 transition-colors hover:border-line-strong md:p-6"
          >
            <h3 className="max-w-[26ch] text-h3 font-semibold leading-tight tracking-tight underline decoration-transparent underline-offset-4 group-hover:decoration-line-strong">
              What does a typed decision cost at {jevSmall.n_options} options and at{" "}
              {`${jevWide.n_options}?`}
            </h3>
            <p className="mt-3 max-w-[68ch] text-body leading-relaxed text-ink-soft">
              Nine models classify the same text against the same label list, first{" "}
              {jevSmall.n_options} options wide and then {`${jevWide.n_options}.`} The decision model is the
              fastest hosted one on the slate and its median hardly notices the wider list, and it is
              the most accurate on neither task. The position bias the run was built to measure turns
              up in {movedByOrder} model/task pair out of {typed.position_bias.length}.
            </p>
            <dl className="numeric mt-6 grid grid-cols-2 gap-x-6 gap-y-3 text-micro sm:grid-cols-4">
              <Stat
                term="Jev latency"
                value={`${jevRise.rise >= 0 ? "+" : "−"}${Math.abs(jevRise.rise * 100).toFixed(0)}%`}
                note={`for ${(jevWide.n_options / jevSmall.n_options).toFixed(0)}x the options`}
                up
              />
              <Stat
                term="Tasks Jev leads on"
                value={`${jevAccuracyWins} of ${typed.tasks.length}`}
                note="accuracy, against eight rivals"
                down
              />
              <Stat
                term="Order-sensitive"
                value={`${movedByOrder} of ${typed.position_bias.length}`}
                note="model/task pairs, paired test"
              />
              <Stat
                term="Recorded calls"
                value={typed.n_calls.toLocaleString()}
                note={`$${typed.spend_usd.toFixed(4)} spent`}
              />
            </dl>
          </Link>
        </li>
        <li>
          <Link
            href="/banking77/"
            className="group block border border-line bg-surface p-5 transition-colors hover:border-line-strong md:p-6"
          >
            <h3 className="max-w-[26ch] text-h3 font-semibold leading-tight tracking-tight underline decoration-transparent underline-offset-4 group-hover:decoration-line-strong">
              Does Laya&apos;s Banking77 failure come from its token budget?
            </h3>
            <p className="mt-3 max-w-[68ch] text-body leading-relaxed text-ink-soft">
              The model card blames a shared option budget that leaves {bankingDefault.n_options}{" "}
              intents about four tokens each. Raising it removes every truncation collision and buys
              back {pct(banking.headline.share_of_deficit, 1)} of the gap to the published Jev number,
              then stops buying anything. The documented workaround scored below doing nothing at all.
            </p>
            <dl className="numeric mt-6 grid grid-cols-2 gap-x-6 gap-y-3 text-micro sm:grid-cols-4">
              <Stat
                term="Room buys"
                value={points(banking.headline.budget_gain_points)}
                note="accuracy points, paired"
                up
              />
              <Stat
                term="Still behind"
                value={banking.headline.remaining_points.toFixed(1)}
                note="points, published Jev"
                down
              />
              <Stat
                term="The workaround"
                value={points(banking.headline.workaround_vs_default_points)}
                note="points vs doing nothing"
                down
              />
              <Stat
                term="Test rows"
                value={bankingRaised.n_test.toLocaleString()}
                note={`BANKING77, ${banking.device.toUpperCase()}`}
              />
            </dl>
          </Link>
        </li>
        <li>
          <Link
            href="/rerank/"
            className="group block border border-line bg-surface p-5 transition-colors hover:border-line-strong md:p-6"
          >
            <h3 className="max-w-[26ch] text-h3 font-semibold leading-tight tracking-tight underline decoration-transparent underline-offset-4 group-hover:decoration-line-strong">
              Can a decision model re-rank retrieval better than BM25?
            </h3>
            <p className="mt-3 max-w-[68ch] text-body leading-relaxed text-ink-soft">
              BM25 retrieves {run.top_k} candidates per query. Each one is handed to a model as a single typed
              question, query and passage in, one number out, and the candidates are re-sorted by that number.
              Nothing is trained. The prediction was that the open-weights model would win, and it lost.
            </p>
            <dl className="numeric mt-6 grid grid-cols-2 gap-x-6 gap-y-3 text-micro sm:grid-cols-4">
              <Stat term="Jev vs BM25" value={signed(jev["ndcg@10_vs_bm25"]!.mean)} note="nDCG@10, paired" up />
              <Stat
                term="Laya vs BM25"
                value={signed(laya["ndcg@10_vs_bm25"]!.mean)}
                note="nDCG@10, paired"
                down
              />
              <Stat term="Queries" value={String(run.queries)} note={run.dataset} />
              <Stat
                term="Scoring calls"
                value={examples.records_in_run.toLocaleString()}
                note="request and response kept"
              />
            </dl>
          </Link>
        </li>
      </ul>

      <h2 className="mt-16 border-t border-line pt-8 text-h3 font-semibold tracking-tight">Related</h2>
      <p className="mt-4 max-w-[68ch] text-body leading-relaxed">
        <a
          href="https://arena.codewithnk.com"
          className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          Decision Arena
        </a>{" "}
        is the sibling experiment: the same two models playing highway-env, Snake and Blackjack with zero
        training, every request and answer on the page. This site asks the same models a retrieval question
        instead of a game one.
      </p>
    </main>
  );
}

function Stat({
  term,
  value,
  note,
  up,
  down,
}: {
  term: string;
  value: string;
  note: string;
  up?: boolean;
  down?: boolean;
}) {
  return (
    <div>
      <dt className="text-ink-soft">{term}</dt>
      <dd className="text-lead" style={{ color: up ? "var(--up)" : down ? "var(--down)" : "var(--ink)" }}>
        {value}
      </dd>
      <dd className="text-ink-soft">{note}</dd>
    </div>
  );
}
