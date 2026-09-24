import Link from "next/link";
import results from "@/data/results.json";
import examples from "@/data/examples.json";
import { byMethod, signed, type Results } from "@/lib/results";

const run = results as Results;
const jev = byMethod(run, "jev-score")!;
const laya = byMethod(run, "laya-score")!;

export default function Page() {
  return (
    <main className="mx-auto max-w-[1100px] px-4 pb-20 pt-10 md:px-8">
      <h1 className="max-w-[16ch] text-h1 font-extrabold leading-none tracking-tight">AI Experiments</h1>
      <p className="mt-4 max-w-[62ch] text-lead leading-relaxed text-ink-soft">
        One question at a time, measured on a public benchmark, with the exact request and response kept
        for every call. A result that comes out negative stays negative here.
      </p>

      <h2 className="mt-14 text-h3 font-extrabold tracking-tight">Experiments</h2>
      <ul className="mt-4 grid gap-4">
        <li>
          <Link
            href="/rerank/"
            className="group block border border-line bg-surface p-5 transition-colors hover:border-line-strong md:p-6"
          >
            <p className="text-micro font-semibold uppercase tracking-wide text-ink-soft">
              Retrieval · BEIR NFCorpus · {run.queries} queries
            </p>
            <h3 className="mt-2 text-h3 font-extrabold tracking-tight group-hover:underline decoration-accent decoration-2 underline-offset-4">
              Can a decision model re-rank retrieval better than BM25?
            </h3>
            <p className="mt-3 max-w-[70ch] text-body leading-relaxed text-ink-soft">
              BM25 retrieves {run.top_k} candidates per query. Each one is handed to a model as a single typed
              question — query and passage in, one number out — and the candidates are re-sorted by that number.
              Nothing is trained.
            </p>
            <dl className="numeric mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-micro sm:grid-cols-4">
              <Stat term="Jev vs BM25" value={signed(jev["ndcg@10_vs_bm25"]!.mean)} note="nDCG@10, paired" up />
              <Stat term="Laya vs BM25" value={signed(laya["ndcg@10_vs_bm25"]!.mean)} note="nDCG@10, paired" down />
              <Stat term="Scoring calls" value={examples.records_in_run.toLocaleString()} note="request and response kept" />
              <Stat term="Browsable" value={`${examples.exported.length} queries`} note="a curated subset" />
            </dl>
          </Link>
        </li>
      </ul>

      <h2 className="mt-14 text-h3 font-extrabold tracking-tight">Related</h2>
      <p className="mt-4 max-w-[70ch] text-body leading-relaxed">
        <a
          href="https://arena.codewithnk.com"
          className="font-semibold underline decoration-accent decoration-2 underline-offset-4"
        >
          Decision Arena
        </a>{" "}
        is the sibling experiment: the same two models, Jev and Laya, playing highway-env, Snake and Blackjack
        with zero training, every request and answer on the page. This site asks the same models a retrieval
        question instead of a game one.
      </p>
    </main>
  );
}

function Stat({ term, value, note, up, down }: {
  term: string; value: string; note: string; up?: boolean; down?: boolean;
}) {
  return (
    <div>
      <dt className="font-semibold text-ink-soft">{term}</dt>
      <dd className={`text-lead font-extrabold ${up ? "text-clear" : down ? "text-danger" : "text-ink"}`}>{value}</dd>
      <dd className="text-ink-soft">{note}</dd>
    </div>
  );
}
