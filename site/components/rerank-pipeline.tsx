import { Term } from "@/components/term";
import { floorRow, movableSplit, rerankerRows, runFacts, type Rerank } from "@/lib/rerank";

// The protocol's worked example, so the page and the README tell one story.
const EXAMPLE = "PLAIN-102";

interface Stage {
  name: string;
  measure: string;
  says: React.ReactNode;
}

/**
 * The experiment as the path one question takes through it, which is also the
 * shortest honest explanation of re-ranking: a cheap search narrows thousands of
 * documents to a shortlist, a careful model reorders the shortlist, and people's
 * judgements grade the order. Every count is read from the run.
 */
export function RerankPipeline({ data }: { data: Rerank }) {
  const facts = runFacts(data);
  const split = movableSplit(data);
  const methods = rerankerRows(data).length;
  const floor = floorRow(data)!["ndcg@10"];
  const config = data.config as { encoder: string; rrf_k: number; depth: number };
  const example = data.queries.find((q) => q.id === EXAMPLE);
  const encoder = config.encoder.split("/").pop();

  const stages: Stage[] = [
    {
      name: "Questions",
      measure: `${facts.queries} questions · ${facts.documents?.toLocaleString() ?? "?"} abstracts`,
      says: (
        <>
          Health questions from NutritionFacts.org and the PubMed abstracts that might answer them, from{" "}
          <Term id="nfcorpus">BEIR NFCorpus</Term>. People have already marked which abstracts answer which
          question
          {example ? (
            <>
              , for instance <q className="italic">{example.text}</q>
            </>
          ) : null}
          .
        </>
      ),
    },
    {
      name: "Retrieve",
      measure: `BM25 + ${encoder} · top ${config.depth} each`,
      says: (
        <>
          Two cheap searches run over every abstract. <Term id="bm25">One matches words</Term>,{" "}
          <Term id="encoder">the other matches meaning</Term>, and their two rankings are{" "}
          <Term id="rrf">fused into one</Term> (k = {config.rrf_k}). Fast, and rough about order.
        </>
      ),
    },
    {
      name: "Shortlist",
      measure: `top ${facts.topK} per question · frozen`,
      says: (
        <>
          The best {facts.topK} are kept and pinned, so every re-ranker sees the same passages. For{" "}
          {split.excluded} questions none of the {facts.topK} is relevant, and nothing after this point can
          help them.
        </>
      ),
    },
    {
      name: "Re-rank",
      measure: `${methods} models · ${facts.calls.toLocaleString()} calls`,
      says: (
        <>
          Each model reads the question beside one passage at a time and scores it; the {facts.topK} are
          sorted by score. A re-ranker can change the order. It can never add a passage the search missed.
        </>
      ),
    },
    {
      name: "Grade",
      measure: `nDCG@10 · floor ${floor.toFixed(3)}`,
      says: (
        <>
          The new order is <Term id="ndcg">scored</Term> against the human judgements: higher when relevant
          abstracts sit nearer the top. The search&apos;s own order, untouched, is{" "}
          <Term id="floor">the floor</Term> every re-ranker has to beat.
        </>
      ),
    },
  ];

  return (
    <ol className="relative grid gap-10 lg:grid-cols-5 lg:gap-6">
      <span aria-hidden className="absolute bottom-2 left-[5px] top-2 w-px bg-line-strong lg:hidden" />
      <span aria-hidden className="absolute left-0 right-0 top-[5px] hidden h-px bg-line-strong lg:block" />
      {stages.map((stage) => (
        <li key={stage.name} className="relative grid content-start gap-2 pl-8 lg:pl-0 lg:pt-8">
          <span
            aria-hidden
            className="absolute left-0 top-1.5 size-[11px] rounded-full border-2 border-ink bg-page lg:top-0"
          />
          <h3 className="text-lead font-semibold leading-tight">{stage.name}</h3>
          <p className="numeric text-micro text-ink">{stage.measure}</p>
          <p className="text-micro leading-relaxed text-ink-soft">{stage.says}</p>
        </li>
      ))}
    </ol>
  );
}
