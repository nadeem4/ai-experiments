import type { Metadata } from "next";
import Link from "next/link";
import examplesIndex from "@/data/examples.json";
import pilotData from "@/data/pilot.json";
import resultsData from "@/data/results.json";
import { Caveats } from "@/components/caveats";
import { Mono, Section } from "@/components/chip";
import { Distributions } from "@/components/distributions";
import { ResultsTable } from "@/components/results-table";
import { WorkedExample } from "@/components/worked-example";
import type { ExamplesIndex } from "@/lib/example";
import { byMethod, signed, type Results } from "@/lib/results";

const results = resultsData as Results;
const pilot = pilotData as Results;
const index = examplesIndex as ExamplesIndex;

const jev = byMethod(results, "jev-score")!;
const laya = byMethod(results, "laya-score")!;

const DESCRIPTION =
  `Does a decision model re-rank retrieval better than BM25? On all ${results.queries} BEIR NFCorpus test ` +
  `queries: Jev ${signed(jev["ndcg@10_vs_bm25"]!.mean)} nDCG@10, Laya ${signed(laya["ndcg@10_vs_bm25"]!.mean)}.`;

export const metadata: Metadata = {
  title: "Re-ranking retrieval with a decision model",
  description: DESCRIPTION,
  openGraph: { title: "Re-ranking retrieval with a decision model", description: DESCRIPTION },
};

export default function Page() {
  return (
    <main className="mx-auto max-w-[1100px] px-4 pb-24 pt-10 md:px-8">
      <p className="text-micro font-semibold uppercase tracking-wide text-ink-soft">
        Experiment 01 · {results.dataset} · {results.queries} queries
      </p>
      <h1 className="mt-2 max-w-[20ch] text-h1 font-extrabold leading-none tracking-tight">
        Re-ranking retrieval with a decision model
      </h1>
      <p className="mt-4 max-w-[64ch] text-lead leading-relaxed text-ink-soft">
        BM25 retrieves {results.top_k} candidate passages per query. Each candidate is handed to a model as one
        typed question — query and passage in, one number out — and the candidates are re-sorted by that number.
        Does the ranking get better?
      </p>
      <p className="mt-4 max-w-[64ch] text-body leading-relaxed">
        Nothing is trained here. This is deliberately zero-shot, which is the baseline that would make a later
        fine-tune interpretable. Four re-rankers were measured against the same BM25 floor:{" "}
        <b>Jev</b> over HTTP, a <b>MS MARCO cross-encoder</b> locally, and <b>Laya</b> on two checkpoints.
      </p>
      <p className="mt-4 max-w-[64ch] border-l-2 border-line-strong pl-4 text-body leading-relaxed">
        <b>Everything on this page comes from the run&apos;s own output.</b> The table below is all{" "}
        {results.queries} queries. The browsable examples further down are a curated subset of{" "}
        {index.exported.length} of them — the full wire is {index.records_in_run.toLocaleString()} recorded
        calls, far too much to send to a browser. The examples are a subset; the numbers are not.
      </p>

      <Section
        id="finding"
        n={1}
        title="The finding"
        standfirst={
          <>
            Jev and the cross-encoder re-rank better than BM25. Both Laya checkpoints re-rank{" "}
            <b className="text-ink">worse</b> than doing nothing at all.
          </>
        }
      >
        <ResultsTable results={results} />
        <ul className="grid max-w-[70ch] gap-3 text-body leading-relaxed">
          <li className="border-l-2 border-line pl-4">
            <b>Jev re-ranks better than BM25</b>, {signed(jev["ndcg@10_vs_bm25"]!.mean)} nDCG@10, better on{" "}
            {jev["ndcg@10_vs_bm25"]!.better} queries and worse on {jev["ndcg@10_vs_bm25"]!.worse}. The gateway
            charged ${jev.calls!.market_cost_usd.toFixed(4)} for the whole pass.
          </li>
          <li className="border-l-2 border-line pl-4">
            <b>Both Laya checkpoints re-rank worse than BM25.</b> Zero-shot, on this dataset, with these
            questions, re-ranking with Laya is worse than leaving the BM25 order alone. Laya&apos;s own model
            card says its base checkpoints are near chance on typed decisions zero-shot, so this is expected
            information rather than a bug — but it is still a loss, and the <Mono>typed-decisions</Mono>{" "}
            fine-tune is the worse of the two.
          </li>
        </ul>
      </Section>

      <Section
        id="worked"
        n={2}
        title="One query, worked through"
        standfirst={
          <>
            BM25&apos;s ordering of its {results.top_k} candidates stays on the left on every view, because it
            is the floor everything is measured against. Pick a method to put beside it, and click any passage
            in the right-hand column for the exact call behind it.
          </>
        }
      >
        <WorkedExample index={index} />
      </Section>

      <Section
        id="distribution"
        n={3}
        title="What the scores look like"
        standfirst="Two of the findings are easier to see in the shape of the scores than in the ranking metric."
      >
        <Distributions results={results} />
      </Section>

      <Section
        id="caveats"
        n={4}
        title="What this does not establish"
        standfirst="Four things the run leaves open, stated here rather than in a footnote."
      >
        <Caveats results={results} pilot={pilot} />
        <p className="max-w-[70ch] text-body leading-relaxed text-ink-soft">
          The repository&apos;s README carries the rest: why the fine-tune loses, whether a better question
          helps, whether the 1,000-character passage cut matters, and what any of this costs on a GPU.
        </p>
      </Section>

      <p className="mt-16 border-t border-line pt-4 text-micro text-ink-soft">
        <Link href="/" className="underline decoration-accent decoration-2 underline-offset-4">
          All experiments
        </Link>{" "}
        ·{" "}
        <a
          href="https://github.com/nadeem4/ai-experiments"
          className="underline decoration-accent decoration-2 underline-offset-4"
        >
          The code, the results files and the tests
        </a>{" "}
        ·{" "}
        <a
          href="https://arena.codewithnk.com"
          className="underline decoration-accent decoration-2 underline-offset-4"
        >
          Decision Arena, the sibling experiment
        </a>
      </p>
    </main>
  );
}
