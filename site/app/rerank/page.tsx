import type { Metadata } from "next";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import Link from "next/link";
import { isPublished, requirePublished } from "@/lib/published";
import rerankData from "@/data/rerank.json";
import { RerankExplorer } from "@/components/rerank-explorer";
import { RerankStoryExperiment } from "@/components/rerank-story-experiment";
import { RerankStoryFound } from "@/components/rerank-story-found";
import { RerankStoryIdea } from "@/components/rerank-story-idea";
import { RerankStoryProblem } from "@/components/rerank-story-problem";
import { floorRow, label, movableSplit, rerankerRows, runFacts, type Detail, type Rerank } from "@/lib/rerank";
import { ordinal, storyQuestion } from "@/lib/rerank-story";

const data = rerankData as Rerank;
const facts = runFacts(data);
const split = movableSplit(data);
const rows = rerankerRows(data);
const best = rows[0];
const floor = floorRow(data)!;
const jev = rows.find((row) => row.method === "jev-score")!;
const YARDSTICK = "cross-encoder";

// The 1.4MB companion is read here, at build time, for the one question the
// story follows; only that question's 20 titles and two orders reach the browser.
const detail = JSON.parse(readFileSync(join(process.cwd(), "public/rerank-detail.json"), "utf8")) as Detail;
const story = storyQuestion(data, detail);

const secondsPerQuestion = (method: string) =>
  data.summary.methods.find((m) => m.method === method)!.scoring_wall_clock_s / facts.queries;
const costOf = (method: string) => data.summary.methods.find((m) => m.method === method)!.calls?.market_cost_usd ?? 0;
const labels = Object.fromEntries(rows.map((row) => [row.method, row.label]));
const REPO = "https://github.com/nadeem4/ai-experiments";

const TITLE = "Can a decision model re-rank retrieval better than hybrid search?";
const DESCRIPTION =
  `A search found the right abstract and put it ${ordinal(story.searchAt)}. We gave four re-rankers the same ` +
  `${facts.queries} health questions: ${best.label} lifted nDCG@10 from ${floor["ndcg@10"].toFixed(3)} to ` +
  `${best.ndcg.toFixed(3)}, and both Laya models made the order worse than doing nothing.`;

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
    <main className="mx-auto max-w-[1180px] px-4 pb-28 pt-6 md:px-8">
      <RerankStoryProblem
        story={story}
        documents={facts.documents ?? 0}
        questions={facts.queries}
        rerankers={rows.length}
      />
      <RerankStoryIdea story={story} />
      <RerankStoryExperiment story={story} labels={labels} questions={facts.queries} floor={floor["ndcg@10"]} />
      <RerankStoryFound yardstick={YARDSTICK} leader={best.method}>
        <div className="grid max-w-[46rem] gap-2.5 text-lead">
          <h3 className="font-semibold">The catch</h3>
          <p className="text-ink-soft">
            {jev.label} is the only one that runs on someone else&apos;s servers, and it is by far the slowest.
            Re-ranking one question means {facts.topK} calls, and made one after another they took about{" "}
            <b className="text-ink">{secondsPerQuestion(jev.method).toFixed(1)} seconds</b>, against{" "}
            {secondsPerQuestion(YARDSTICK).toFixed(2)} for the {label(YARDSTICK).toLowerCase()}. Whether the calls can
            run at once was not tested.
          </p>
          <table className="w-full max-w-[34rem] border-collapse text-[0.9375rem]">
            <thead>
              <tr className="border-b border-line text-left text-micro text-ink-soft">
                <th className="py-2 pr-2.5 font-normal">Re-ranker</th>
                <th className="py-2 pr-2.5 text-right font-normal">Seconds per question</th>
                <th className="py-2 text-right font-normal">Cost for all {facts.queries}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.method} className="border-b border-line">
                  <td className="py-2 pr-2.5">{row.label}</td>
                  <td className="numeric py-2 pr-2.5 text-right">{secondsPerQuestion(row.method).toFixed(2)}</td>
                  <td className="numeric py-2 text-right">
                    {costOf(row.method) ? `$${costOf(row.method).toFixed(2)}` : "free"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-ink-soft">
            {jev.label}&apos;s time is a trip across the internet; the other three ran on one local graphics card. The
            cost is what the provider charged.
          </p>
        </div>
      </RerankStoryFound>

      <RerankExplorer />

      <section aria-labelledby="limits" className="mt-20 grid gap-5 border-t border-line pt-16">
        <h2 id="limits" className="text-h2 font-semibold leading-tight tracking-tight">
          What this does not answer.
        </h2>
        <p className="max-w-[46rem] text-lead text-ink-soft">
          The numbers above are easy to read too much into. Each of these is a question the run cannot settle.
        </p>
        <ul className="grid max-w-[52rem]">
          {[
            [
              `Whether ${jev.label} is fast enough for someone waiting on a search.`,
              `One question is ${facts.topK} calls, about ${secondsPerQuestion(jev.method).toFixed(1)} seconds made one after another. Whether they can run at once, and how fast that would be, was not tested.`,
            ],
            [
              `Whether ${jev.label} beats a bigger cross-encoder.`,
              "The one here is the smallest common model, 22M parameters. A team choosing for real would try a larger one.",
            ],
            [
              `How much of ${jev.label}'s order is really ${jev.label}'s.`,
              `Its scores come back with two decimals, so ${jev.scores!.tied.toLocaleString("en-US")} of the ${jev.scores!.of.toLocaleString("en-US")} abstracts it scored tie with another, and a tie keeps the search's order.`,
            ],
            [
              "Whether training would rescue Laya.",
              "Neither Laya was trained on this task here. Their numbers are the starting point a later fine-tune would be measured against.",
            ],
            [
              "Anything the search never found.",
              `On ${split.excluded} of the ${facts.queries} questions, none of the ${facts.topK} abstracts was a right answer, so no re-ranker could help.`,
            ],
            ["Whether it holds beyond health questions.", "One collection of medical abstracts, one kind of question."],
          ].map(([head, body]) => (
            <li key={head} className="grid gap-1 border-t border-line py-3.5 last:border-b">
              <b className="text-[1.05rem]">{head}</b>
              <span className="text-[0.96875rem] leading-normal text-ink-soft">{body}</span>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="record" className="mt-20 grid gap-5 border-t border-line pt-16">
        <h2 id="record" className="text-h2 font-semibold leading-tight tracking-tight">
          The full record.
        </h2>
        <div className="grid gap-2.5 md:grid-cols-4">
          {[
            [`${REPO}/blob/main/rerank/README.md`, "The plan", "What was tested and how, with every change made since the run dated."],
            [`${REPO}/blob/main/rerank/RESULTS.md`, "Every result", "All eight measures with their ranges, every pair compared, cost and speed."],
            [
              "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
              "The questions",
              `NFCorpus: ${(facts.documents ?? 0).toLocaleString("en-US")} medical abstracts and ${facts.queries} test questions, from the University of Heidelberg.`,
            ],
            [`${REPO}/tree/main/rerank`, "The code", "Everything needed to run it again, and the tables every number here comes from."],
          ].map(([href, head, body]) => (
            <a key={head} href={href} className="grid content-start gap-1 border border-line bg-surface px-4 py-3.5 hover:border-ink">
              <b className="text-[1.05rem]">{head}</b>
              <span className="text-sm leading-normal text-ink-soft">{body}</span>
            </a>
          ))}
        </div>
      </section>

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
          </Link>
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
