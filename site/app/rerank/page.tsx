import type { Metadata } from "next";
import Link from "next/link";
import rerankData from "@/data/rerank.json";
import { Mono, Prose, Section } from "@/components/chip";
import { RerankExplorer } from "@/components/rerank-explorer";
import { RerankIntervals } from "@/components/rerank-intervals";
import { movableSplit, pct, rerankerRows, runFacts, signed, type Rerank } from "@/lib/rerank";

const data = rerankData as Rerank;
const facts = runFacts(data);
const split = movableSplit(data);
const rows = rerankerRows(data);

const best = rows[0];
const worst = rows[rows.length - 1];
const jev = rows.find((row) => row.method === "jev-score")!;
const ce = rows.find((row) => row.method === "cross-encoder")!;
const local = rows.filter((row) => row.method !== "jev-score");
const fastestLocal = local.reduce((a, b) => ((a.p50 ?? Infinity) < (b.p50 ?? Infinity) ? a : b));
const slowestLocal = local.reduce((a, b) => ((a.p50 ?? 0) > (b.p50 ?? 0) ? a : b));

const config = data.config as {
  encoder: string;
  fusion: string;
  rrf_k: number;
  depth: number;
  first_stage: string;
};

const TITLE = "Can a decision model re-rank retrieval better than hybrid search?";
const DESCRIPTION =
  `${facts.queries} health questions, ${facts.topK} hybrid candidates each, four re-rankers scoring every ` +
  `pair one at a time. Jev ${signed(best.mean)} nDCG@10 against the ${data.floor} floor, ` +
  `${ce.method} ${signed(ce.mean)}, and both Laya checkpoints negative.`;

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION },
};

export default function Page() {
  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-28 pt-12 md:px-8">
      <h1 className="max-w-[18ch] text-h1 font-semibold leading-[1.05] tracking-tight">{TITLE}</h1>
      <p className="mt-6 max-w-[64ch] text-lead leading-relaxed text-ink-soft">
        Zero is the order the retriever already gave you. Everything to the right of it is a re-ranker
        earning its place in the pipeline, and everything to the left is one that should be deleted from it.
      </p>

      <RerankIntervals data={data} />

      <dl className="numeric mt-8 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-6 text-micro sm:grid-cols-3 lg:grid-cols-6">
        <Fact term="Scoring calls" value={facts.calls.toLocaleString()} note={`${facts.failed} failed`} />
        <Fact term="Questions" value={String(facts.queries)} note={`${split.movable} could move`} />
        <Fact term="Candidates each" value={String(facts.topK)} note={`${facts.passages.toLocaleString()} distinct abstracts`} />
        <Fact term="Spend" value={`$${facts.spendUsd.toFixed(2)}`} note={`$${facts.spendUsd.toFixed(4)} exactly`} />
        <Fact term="Device" value={facts.device} note={`commit ${data.summary.commit ?? "unrecorded"}`} />
        <Fact
          term="Finished"
          value={data.summary.finished.slice(0, 10)}
          note={facts.dataset}
        />
      </dl>

      <p className="mt-6 max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        <b>One of these four is not measured like the others.</b> <Mono>{jev.method}</Mono> answers over the
        network, so its p50 of {jev.p50} ms is a round trip and not a forward pass. The three local scorers
        ran on the same device — {fastestLocal.method} at {fastestLocal.p50} ms, {slowestLocal.method} at{" "}
        {slowestLocal.p50} ms — and are comparable with each other. Putting {jev.method} on that axis would
        be measuring the distance to a server.
      </p>

      <Section
        id="run"
        title="What the run was"
        standfirst={`${facts.queries} health questions against a corpus of medical abstracts, re-ranked four ways over the same pinned candidate lists.`}
      >
        <Prose>
          <p>
            The first stage is hybrid: BM25 and <Mono>{config.encoder}</Mono> each run to depth{" "}
            {config.depth} and their two rankings are fused by {config.fusion.replace("-", " ")} at k ={" "}
            {config.rrf_k}, keeping the top {facts.topK}. That fused order is the <Mono>{data.floor}</Mono>{" "}
            row, and it is the floor: it is what you get for doing nothing after retrieval, and it is the
            line every other row on this page is measured from rather than a competitor in the table.
          </p>
          <p>
            Each re-ranker then sees one (question, passage) pair at a time — {facts.topK} pairs per
            question, {facts.calls.toLocaleString()} in all — and returns a single number. The{" "}
            {facts.topK} candidates are sorted by that number and nothing else changes: same documents, same
            question, a different order. Nothing was trained and no prompt was tuned against the test split.
          </p>
          <p>
            {split.excluded} of the {facts.queries} questions have nothing relevant among their{" "}
            {facts.topK} candidates. No re-ranker can score above zero on those whatever it does, and they
            are carried in the means but left out of the per-question charts, where they would be{" "}
            {split.excluded} bars of nothing.
          </p>
        </Prose>
      </Section>

      <RerankExplorer />

      <Section
        id="not-answered"
        title="What this does not answer"
        standfirst="Five things the run is silent on. They are here because the numbers above are easy to over-read, and each of these is a reading they do not support."
      >
        <Prose>
          <p>
            <b>Whether {jev.method} beats {ce.method}.</b> Each interval is against the common floor, and two
            intervals that both exclude zero do not establish an ordering between them. The paired{" "}
            {jev.method}-minus-{ce.method} difference would, and it was not computed.
          </p>
          <p>
            <b>What {jev.method}&apos;s ties are worth.</b> Its answers land on a coarse grid —{" "}
            {jev.scores!.distinct} distinct values across {jev.scores!.of.toLocaleString()} scored passages,
            the largest single tie {jev.scores!.largest_group} of {facts.topK} candidates in one question —
            and a tie keeps the retriever&apos;s order. So {pct(jev.scores!.tied / jev.scores!.of)} of the
            passages credited to it are holding the first stage&apos;s ranking rather than expressing one.
            That cuts both ways and this run does not separate them.
          </p>
          <p>
            <b>Whether a fine-tune would rescue Laya.</b> This is zero-shot by design. {worst.method} at{" "}
            {signed(worst.mean)} is the baseline that would make a later fine-tune interpretable, not a
            verdict on the architecture.
          </p>
          <p>
            <b>Anything beyond {facts.topK} candidates.</b> Recall@10 is bounded by what the first stage
            retrieved: the floor found {(data.summary.methods.find((m) => m.method === data.floor)!["recall@10"] * 100).toFixed(1)}% of
            the relevant passages and {jev.method} pulled that to {(jev.recall * 100).toFixed(1)}%, but a
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

function Fact({ term, value, note }: { term: string; value: string; note: string }) {
  return (
    <div>
      <dt className="text-ink-soft">{term}</dt>
      <dd className="text-lead text-ink">{value}</dd>
      <dd className="text-ink-soft">{note}</dd>
    </div>
  );
}
