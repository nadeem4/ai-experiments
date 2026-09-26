import type { Metadata } from "next";
import Link from "next/link";
import examplesIndex from "@/data/examples.json";
import perQueryData from "@/data/per-query.json";
import pilotData from "@/data/pilot.json";
import questionData from "@/data/question.json";
import resultsData from "@/data/results.json";
import scoresData from "@/data/scores.json";
import { Assumptions } from "@/components/assumptions";
import { Mono, Prose, Section } from "@/components/chip";
import { Code } from "@/components/code";
import { Downloads } from "@/components/downloads";
import { Explorer } from "@/components/explorer";
import { IntervalPlot } from "@/components/interval-plot";
import { QuantisationRug } from "@/components/quantisation-rug";
import { ResultsTable } from "@/components/results-table";
import type { ExamplesIndex } from "@/lib/example";
import type { PerQuery } from "@/lib/per-query";
import { byMethod, pct, signed, tieShare, type Results } from "@/lib/results";

const results = resultsData as Results;
const pilot = pilotData as Results;
const index = examplesIndex as ExamplesIndex;
const perQuery = perQueryData as PerQuery;
const scores = scoresData as Record<string, { of: number; values: number[] }>;

const jev = byMethod(results, "jev-score")!;
const ce = byMethod(results, "cross-encoder")!;
const laya = byMethod(results, "laya-score")!;
const tuned = byMethod(results, "laya-typed-score")!;

const means = Object.fromEntries(
  perQuery.methods.map((method) => [method, byMethod(results, method)!["ndcg@10_vs_bm25"]!.mean]),
);

const DESCRIPTION =
  `Can a decision model re-rank retrieval better than BM25? On all ${results.queries} BEIR NFCorpus test ` +
  `queries: Jev ${signed(jev["ndcg@10_vs_bm25"]!.mean)} nDCG@10, Laya ${signed(laya["ndcg@10_vs_bm25"]!.mean)}. ` +
  `The prediction was the other way round.`;

const TITLE = "Can a decision model re-rank retrieval better than BM25?";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION },
};

export default function Page() {
  const unjudged = perQuery.rows.filter((row) => row.relevant === 0).length;

  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-28 pt-12 md:px-8">
      <h1 className="max-w-[17ch] text-h1 font-semibold leading-[1.05] tracking-tight">{TITLE}</h1>
      <p className="numeric mt-6 max-w-[62ch] text-micro text-ink-soft">
        {results.dataset}, all {results.queries} test queries, top {results.top_k} candidates,{" "}
        {index.records_in_run.toLocaleString()} recorded calls, nothing trained.
      </p>

      <Section
        id="why"
        title="Why it is worth asking"
        standfirst="Re-ranking is a real stage in a real pipeline, and today it is served either by a cross-encoder or by a language model call per passage."
      >
        <Prose>
          <p>
            A decision model does not write text. You hand it the situation in words and the answers you will
            accept, and it hands back a number: a probability, or a position on a scale you defined. There is
            nothing to parse and it cannot answer off the list. If one can judge relevance in tens of
            milliseconds, it is cheaper to operate than a cross-encoder and far cheaper than a language model
            call per candidate. That is the case worth testing.
          </p>
          <p>
            There is also a prior. An{" "}
            <a
              href="https://arena.codewithnk.com"
              className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
            >
              earlier experiment
            </a>{" "}
            put the same two models in front of games, and Laya, the open-weights one, failed badly at them.
            Games are a long way outside what a relevance model was built for, so that result proved little
            about the model. Retrieval relevance is its home domain. If it is going to work anywhere, it is
            here, which makes this the fair test rather than the easy one.
          </p>
        </Prose>
      </Section>

      <Section
        id="expected"
        title="What we expected"
        standfirst="Written up after the run, not before it. The reasoning is real and it was wrong, but it is reasoning rather than a pre-registered prediction, and this page should not imply otherwise."
      >
        <div className="max-w-[62ch] border border-line bg-surface p-5">
          <p className="numeric text-micro text-ink-soft">What we believed going in, recorded afterwards</p>
          <p className="mt-3 text-lead italic leading-snug">
            Laya will re-rank well here, and better than Jev, because Convai&apos;s published benchmarks put it
            ahead of Jev on text relevance tasks, and this task is text relevance.
          </p>
        </div>
        <Prose>
          <p>
            It was wrong in both halves. Laya did not re-rank well: both of its checkpoints left the ranking
            worse than doing nothing. And it was not ahead of Jev, which turned out to be the method that moved
            the ranking furthest in the run.
          </p>
          <p>
            Laya&apos;s own model card does say its base checkpoints sit near chance on typed decisions
            zero-shot, so the size of the loss is expected information rather than a defect. The reasoning
            above is real, and it was written down here only after the answer was known, which is why neither
            this page nor the results file reports a verdict on it. Reasoning recalled afterwards cannot be
            scored against a result it already knows.
          </p>
        </Prose>
      </Section>

      <Section
        id="did"
        title="What we did"
        standfirst={`BM25 retrieves ${results.top_k} candidate passages per query. Each candidate becomes one typed question, and the candidates are re-sorted by the number that comes back.`}
      >
        <Prose>
          <p>
            Every method re-ranks the same candidate list, so the difference is paired per query and BM25 is
            the floor rather than a rival. Four re-rankers were measured against it: <b>Jev</b>, TypeSafe&apos;s
            hosted decision model, over HTTP; a <b>MS MARCO cross-encoder</b> locally; and <b>Laya</b>,
            Convai&apos;s open-weights alternative, on its base checkpoint and on its{" "}
            <Mono>typed-decisions</Mono> fine-tune. Ties keep BM25&apos;s order, so a model with no opinion
            changes nothing rather than shuffling.
          </p>
          <p>
            This is the question, exactly as it went out {index.records_in_run.toLocaleString()} times. It is
            one line of the run log, cut only where the passage runs on.
          </p>
        </Prose>
        <div className="min-w-0 max-w-[78ch]">
          <Code value={questionData.request} className="max-h-none" />
        </div>
        <Prose>
          <p>
            Back comes one number between 0 and 4, and the candidates are sorted by it. Nothing is trained, and
            no prompt was tuned against the test split, which would have made the number meaningless. Zero-shot
            is the baseline that would make a later fine-tune interpretable.
          </p>
        </Prose>
      </Section>

      <Section
        id="happened"
        title="What happened"
        standfirst="Jev and the cross-encoder re-rank better than BM25. Both Laya checkpoints re-rank worse than doing nothing at all."
      >
        <Explorer
          perQuery={perQuery}
          methods={perQuery.methods}
          means={means}
          records={index.records_in_run}
        />

        <ResultsTable results={results} />

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">Four differences, one floor</h3>
          <Prose>
            <p>
              A mean is a result only when the interval around it stays on one side of zero. All four do, in
              both directions, which is what a {results.queries}-query run was for: at the {pilot.queries}-query
              pilot, only Laya&apos;s loss was separable from noise.
            </p>
          </Prose>
          <IntervalPlot results={results} />
        </div>
      </Section>

      <Section
        id="assumed"
        title="What we assumed"
        standfirst="Six boundaries on the result. They are here rather than in a footnote because each one changes what the numbers above are allowed to mean."
      >
        <Assumptions results={results} pilot={pilot} perQuery={perQuery} records={index.records_in_run} />
      </Section>

      <Section
        id="understood"
        title="What we understood"
        standfirst="The answer, and the two things in the run that nobody predicted, including us."
      >
        <Prose>
          <p>
            <b>A decision model can re-rank retrieval better than BM25.</b> Jev moved nDCG@10 by{" "}
            {signed(jev["ndcg@10_vs_bm25"]!.mean)} over all {results.queries} queries, better on{" "}
            {jev["ndcg@10_vs_bm25"]!.better} of them and worse on {jev["ndcg@10_vs_bm25"]!.worse}, and the
            gateway charged ${jev.calls!.market_cost_usd.toFixed(4)} for the whole pass. The cross-encoder
            moved it {signed(ce["ndcg@10_vs_bm25"]!.mean)}. Which of those two is better is not something this
            run can say.
          </p>
          <p>
            <b>The open-weights model made retrieval worse.</b> Laya&apos;s base checkpoint cost{" "}
            {signed(laya["ndcg@10_vs_bm25"]!.mean)} nDCG@10 and its fine-tune cost{" "}
            {signed(tuned["ndcg@10_vs_bm25"]!.mean)}. On this dataset, with these questions, zero-shot, the
            ranking is better if you do not call it. That is the answer the prior set up, and it is the
            opposite of the prediction.
          </p>
        </Prose>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            The first surprise: the fine-tune ranks below the checkpoint it came from
          </h3>
          <Prose>
            <p>
              <Mono>typed-decisions</Mono> is fine-tuned for exactly this kind of question, and it ranks below
              its own base, which contradicts its model card. It is genuinely the checkpoint being loaded:
              different file, different config, and all {laya.scores!.of.toLocaleString()} pairs scored
              differently between the two, not one identical value. What it does instead is use less of the
              scale. The base checkpoint spreads its answers from {laya.scores!.min} to {laya.scores!.max},
              standard deviation {laya.scores!.stdev}; the fine-tune from {tuned.scores!.min} to{" "}
              {tuned.scores!.max}, standard deviation {tuned.scores!.stdev}, never touching the bottom third. A
              re-ranker that separates candidates less ranks them worse. Why fine-tuning did that is not
              answered here.
            </p>
          </Prose>
        </div>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            The second surprise: {pct(tieShare(jev.scores))} of Jev&apos;s passages are ties, and a tie is
            BM25&apos;s answer
          </h3>
          <Prose>
            <p>
              The gateway returns Jev&apos;s answers rounded to two decimals, so a 0-to-4 score has at most 401
              places to land, and Jev used {jev.scores!.distinct} of them across{" "}
              {jev.scores!.of.toLocaleString()} successful calls. The cross-encoder emits a raw logit and
              returned {ce.scores!.distinct.toLocaleString()} distinct values across{" "}
              {ce.scores!.of.toLocaleString()}. Coarse steps mean ties, and tied passages keep the order BM25
              gave them, so {jev.scores!.tied.toLocaleString()} of Jev&apos;s{" "}
              {jev.scores!.of.toLocaleString()} scored passages are holding BM25&apos;s ranking rather than
              expressing one. Its largest single tie was {jev.scores!.largest_group} of {results.top_k}{" "}
              candidates in one query.
            </p>
          </Prose>
          <QuantisationRug
            strips={[
              { method: "jev-score", scores: jev.scores!, values: scores["jev-score"].values },
              { method: "cross-encoder", scores: ce.scores!, values: scores["cross-encoder"].values },
            ]}
          />
          <Prose>
            <p>
              So Jev&apos;s {signed(jev["ndcg@10_vs_bm25"]!.mean)} is earned while roughly a quarter of its
              ranking is still BM25&apos;s. That cuts both ways, and this run does not separate them: it may
              mean Jev is decisive exactly where it matters, or it may mean the gain comes from fewer decisions
              than the call count suggests. The rounding is not something a client can switch off.
            </p>
            <p>
              What would settle the most from here is the comparison that was not run: Jev against the
              cross-encoder, paired, on the same {results.queries} queries. After that, whether finer
              resolution buys anything, and whether {unjudged} queries with nothing relevant in the candidate
              pool are telling us more about NFCorpus than about re-ranking.
            </p>
          </Prose>
        </div>
      </Section>

      <Section
        id="downloads"
        title="The files behind the page"
        standfirst="Everything above was read out of these. Nothing on this site is illustrative."
      >
        <Downloads perQuery={perQuery} records={index.records_in_run} />
      </Section>

      <p className="numeric mt-20 border-t border-line pt-5 text-micro text-ink-soft">
        <Link href="/" className="underline decoration-line-strong underline-offset-4 hover:decoration-ink">
          All experiments
        </Link>{" "}
        ·{" "}
        <a
          href="https://github.com/nadeem4/ai-experiments"
          className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          The code, the results files and the tests
        </a>
      </p>
    </main>
  );
}
