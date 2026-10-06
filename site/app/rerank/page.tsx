import type { Metadata } from "next";
import { isPublished, requirePublished } from "@/lib/published";
import Link from "next/link";
import rerankData from "@/data/rerank.json";
import { Prose, Section } from "@/components/chip";
import { RerankExplorer } from "@/components/rerank-explorer";
import { RerankIntervals } from "@/components/rerank-intervals";
import { RerankPipeline } from "@/components/rerank-pipeline";
import { RerankStandings } from "@/components/rerank-standings";
import { RerankWalkthrough } from "@/components/rerank-walkthrough";
import { Term } from "@/components/term";
import { METHOD_TERM } from "@/lib/rerank-glossary";
import {
  floorRow,
  headToHead,
  movableSplit,
  pct,
  rerankerRows,
  runFacts,
  signed,
  verdict,
  type Rerank,
} from "@/lib/rerank";

const data = rerankData as Rerank;
const facts = runFacts(data);
const split = movableSplit(data);
const rows = rerankerRows(data);

const best = rows[0];
const worst = rows[rows.length - 1];
const jev = rows.find((row) => row.method === "jev-score")!;
const ce = rows.find((row) => row.method === "cross-encoder")!;
const floor = floorRow(data)!;
const bestVsCe = headToHead(data, best.method, ce.method);
const harmful = rows.filter((row) => verdict(row.mean, row.ci95) === "worse");
const [layaBase, layaTyped] = ["laya-score", "laya-typed-score"].map((m) => rows.find((r) => r.method === m)!);
const layaPair = headToHead(data, layaBase.method, layaTyped.method);
const secondsPerQuery = (row: typeof jev) =>
  data.summary.methods.find((m) => m.method === row.method)!.scoring_wall_clock_s / facts.queries;
const interval = (ci: number[] | null | undefined, digits = 4) =>
  ci ? `[${signed(ci[0], digits)}, ${signed(ci[1], digits)}]` : "";

const TITLE = "Can a decision model re-rank retrieval better than hybrid search?";
const DESCRIPTION =
  `${facts.queries} health questions, ${facts.topK} hybrid candidates each, four re-rankers scoring every ` +
  `pair one at a time. Jev lifts nDCG@10 from ${floor["ndcg@10"].toFixed(3)} to ${best.ndcg.toFixed(3)} and beats ` +
  `a cross-encoder head to head; both Laya checkpoints make the order worse than doing nothing.`;

const PAGE_METADATA: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION },
};

// An unpublished page renders the 404 (experiments.config.ts), and its title
// must not ride along in the head.
export const metadata: Metadata = isPublished("rerank") ? PAGE_METADATA : {};

export default function Page() {
  requirePublished("rerank");

  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-28 pt-12 md:px-8">
      <h1 className="max-w-[18ch] text-h1 font-semibold leading-[1.05] tracking-tight">{TITLE}</h1>
      <p className="mt-6 max-w-[60ch] text-lead leading-relaxed">
        {verdict(best.mean, best.ci95) === "better" ? "Yes, one of them. " : "No. "}
        <b>
          <Term id={METHOD_TERM[best.method]}>{best.label}</Term>
        </b>{" "}
        lifts <Term id="ndcg">nDCG@10</Term> from <span className="numeric">{floor["ndcg@10"].toFixed(3)}</span> to{" "}
        <span className="numeric">{best.ndcg.toFixed(3)}</span>,{" "}
        {pct(best.ndcg / floor["ndcg@10"] - 1, 1)} better than <Term id="floor">the order the search already gave</Term>
        {bestVsCe && verdict(bestVsCe.mean, bestVsCe.ci95) === "better" ? (
          <>
            , and beats the <Term id="cross-encoder">{ce.label}</Term> head to head
          </>
        ) : null}
        .{" "}
        {harmful.length > 0 && (
          <>
            <Term id="laya">Laya</Term>, the open-weights model, makes the order worse than doing nothing
            {harmful.length === 2 ? ", in both checkpoints tested" : ""}.
          </>
        )}
      </p>

      <RerankStandings data={data} />

      <p className="numeric mt-6 text-micro text-ink-soft">
        {facts.calls.toLocaleString()} scoring calls, {facts.failed} failed · ${facts.spendUsd.toFixed(2)} in
        all · {split.excluded} of {facts.queries} questions had nothing relevant to find
      </p>

      <Section
        id="thirty-seconds"
        title="Re-ranking in 30 seconds"
        standfirst="One real question from the run, step by step. Every number is the run's own."
      >
        <RerankWalkthrough />
      </Section>

      <Section
        id="how"
        title="How it was measured"
        standfirst={
          <>
            A <Term id="rerank">re-ranker</Term> is the second, slower half of a search. This is the path every
            one of the {facts.queries} questions took.
          </>
        }
      >
        <RerankPipeline data={data} />
      </Section>

      <Section
        id="sure"
        title="How sure we are"
        standfirst={
          <>
            Each re-ranker against the floor, question by question, with a <Term id="interval">95% interval</Term>.
            A result counts only when its whole interval sits on one side of zero.
          </>
        }
      >
        <RerankIntervals data={data} />
        <Prose>
          {bestVsCe && (
            <p>
              Compared with each other directly, on the same questions, <b>{best.label}</b> is ahead of the{" "}
              {ce.label} by <span className="numeric">{signed(bestVsCe.mean)}</span>{" "}
              <span className="numeric">{interval(bestVsCe.ci95)}</span>: better on {bestVsCe.better} questions,
              worse on {bestVsCe.worse}.
            </p>
          )}
          {layaPair && (
            <p>
              The two Laya checkpoints cannot be told apart. The base is ahead of the {layaTyped.label}{" "}
              fine-tune by <span className="numeric">{signed(layaPair.mean)}</span>{" "}
              <span className="numeric">{interval(layaPair.ci95)}</span>, an interval across zero, and each
              wins on about as many questions as the other ({layaPair.better} and {layaPair.worse}).
            </p>
          )}
        </Prose>
      </Section>

      <RerankExplorer />

      <Section
        id="not-answered"
        title="What this does not answer"
        standfirst="Six things the run is silent on. They are here because the numbers above are easy to over-read, and each of these is a reading they do not support."
      >
        <Prose>
          <p>
            <b>Whether {jev.label} is fast enough to put in front of a person.</b> Re-ranking one question is{" "}
            {facts.topK} calls, and made one after another {jev.label}&apos;s take about{" "}
            <span className="numeric">{secondsPerQuery(jev).toFixed(1)} s</span> against the {ce.label}&apos;s{" "}
            <span className="numeric">{secondsPerQuery(ce).toFixed(2)} s</span>. Whether the calls can run at
            once, and what that does to the round trip, was not tested.
          </p>
          <p>
            <b>Whether {jev.label} beats a larger cross-encoder.</b> The one here is the smallest MiniLM. A
            production team would compare against something bigger, and this run did not.
          </p>
          <p>
            <b>What {jev.label}&apos;s ties are worth.</b> Its answers land on a coarse grid —{" "}
            {jev.scores!.distinct} distinct values across {jev.scores!.of.toLocaleString()} scored passages,
            the largest single tie {jev.scores!.largest_group} of {facts.topK} candidates in one question —
            and a tie keeps the retriever&apos;s order. So {pct(jev.scores!.tied / jev.scores!.of)} of the
            passages credited to it are holding the first stage&apos;s ranking rather than expressing one.
            That cuts both ways and this run does not separate them.
          </p>
          <p>
            <b>Whether a fine-tune would rescue Laya.</b> This is zero-shot by design. {worst.label} at{" "}
            {signed(worst.mean)} is the baseline that would make a later fine-tune interpretable, not a
            verdict on the architecture.
          </p>
          <p>
            <b>Anything beyond {facts.topK} candidates.</b> Recall@10 is bounded by what the first stage
            retrieved: the floor found {(data.summary.methods.find((m) => m.method === data.floor)!["recall@10"] * 100).toFixed(1)}% of
            the relevant passages and {jev.label} pulled that to {(jev.recall * 100).toFixed(1)}%, but a
            re-ranker cannot retrieve what was never in the list, and {split.excluded} questions had nothing
            in theirs.
          </p>
          <p>
            <b>Whether this transfers.</b> One corpus, one domain: medical abstracts against health
            questions, on {facts.dataset}. Nothing here says what any of these four would do to a different
            kind of question.
          </p>
        </Prose>
      </Section>

      <footer className="numeric mt-20 border-t border-line-strong pt-6 text-micro text-ink-soft">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3 lg:grid-cols-6">
          <Fact term="Dataset" value={facts.dataset} />
          <Fact term="Device" value={facts.device} />
          <Fact term="Commit" value={data.summary.commit ?? "unrecorded"} />
          <Fact term="Calls" value={`${facts.calls.toLocaleString()}, ${facts.failed} failed`} />
          <Fact term="Spend" value={`$${facts.spendUsd.toFixed(4)}`} />
          <Fact term="Report written" value={data.summary.finished.slice(0, 10)} />
        </dl>
        <p className="mt-6">
          <Link href="/" className="underline decoration-line-strong underline-offset-4 hover:decoration-ink">
            All experiments
          </Link>{" "}
          ·{" "}
          <a
            href="https://github.com/nadeem4/ai-experiments/tree/main/rerank"
            className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
          >
            The protocol, the results files and the tests
          </a>
        </p>
      </footer>
    </main>
  );
}

function Fact({ term, value }: { term: string; value: string }) {
  return (
    <div>
      <dt>{term}</dt>
      <dd className="text-ink">{value}</dd>
    </div>
  );
}
