import rerankData from "@/data/rerank.json";
import { floorRow, rerankerRows, runFacts, type Rerank } from "@/lib/rerank";

/**
 * What the rerank page's technical words mean, for a reader who has never met
 * them. `says` is the plain definition; `here` is what the term did in this run,
 * read from the results file like every other number on the page; `link` is the
 * term's own source, checked to resolve when it was written.
 */
export interface GlossaryEntry {
  term: string;
  kind: string;
  says: string;
  here?: string;
  link?: { href: string; label: string };
}

const data = rerankData as Rerank;
const facts = runFacts(data);
const floor = floorRow(data)!["ndcg@10"];
const row = (method: string) => rerankerRows(data).find((r) => r.method === method)!;
const jev = row("jev-score");
const ce = row("cross-encoder");
const laya = row("laya-score");
const layaTyped = row("laya-typed-score");
const ms = (n: number | null) => `${n?.toFixed(1)} ms`;

const ENTRIES = {
  rerank: {
    term: "Re-ranking",
    kind: "the second half of a search",
    says:
      "A fast search finds a shortlist; a slower, more careful model then reads the question beside each result, scores it, and the shortlist is sorted by score. A re-ranker can change the order. It can never add a result the search missed.",
    here: `${facts.topK} candidates per question, ${facts.calls.toLocaleString()} scoring calls in all.`,
  },
  jev: {
    term: "Jev",
    kind: "decision model by TypeSafe AI",
    says:
      "A hosted model that answers a question with a number, never with free text. You hand it the situation and a typed question, here 'how relevant, 0 to 4?', and it returns the score. Reached over the internet through OpenRouter.",
    here: `nDCG@10 ${jev.ndcg.toFixed(3)} against the floor's ${floor.toFixed(3)}. ${ms(jev.p50)} a call, a network round trip. $${jev.calls!.market_cost_usd.toFixed(2)} for the whole run.`,
    link: { href: "https://typesafe.ai/blog/introducing-system-one-models-and-jev", label: "TypeSafe AI's announcement" },
  },
  laya: {
    term: "Laya",
    kind: "open-weights decision model",
    says:
      "Answers the same typed questions as Jev, but its weights are public, so it runs on your own machine. Two versions were tested: the base model and a fine-tune called typed-decisions. Neither was trained for this task.",
    here: `nDCG@10 ${laya.ndcg.toFixed(3)} (base) and ${layaTyped.ndcg.toFixed(3)} (typed-decisions), both below the floor's ${floor.toFixed(3)}. ${ms(laya.p50)} a call on a T4 GPU.`,
    link: { href: "https://github.com/NandhaKishorM/laya", label: "Laya on GitHub" },
  },
  "cross-encoder": {
    term: "Cross-encoder MiniLM",
    kind: "the conventional re-ranker",
    says:
      "A small transformer, 22M parameters, that reads the question and the passage together and outputs one relevance score. Trained on MS MARCO: real Bing questions, with the passages people marked as answering them. The usual baseline for re-ranking.",
    here: `nDCG@10 ${ce.ndcg.toFixed(3)}. ${ms(ce.p50)} a call on a T4 GPU, free to run.`,
    link: { href: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2", label: "The model on Hugging Face" },
  },
  encoder: {
    term: "all-MiniLM-L6-v2",
    kind: "text encoder",
    says:
      "Turns a piece of text into 384 numbers whose direction carries its meaning. Two texts can then be compared by angle, which finds a match even when they share no words. It is the meaning half of the search here.",
    link: { href: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2", label: "The model on Hugging Face" },
  },
  bm25: {
    term: "BM25",
    kind: "keyword search score",
    says:
      "The classic way to score a document against a query. More of the query's words in the document means a higher score, rare words count for more than common ones, and long documents are not rewarded just for being long.",
    link: { href: "https://en.wikipedia.org/wiki/Okapi_BM25", label: "BM25 on Wikipedia" },
  },
  rrf: {
    term: "Reciprocal rank fusion",
    kind: "merging two rankings",
    says:
      "Each document earns 1 / (60 + its rank) from every list it appears in, and the totals decide the merged order. It uses positions rather than raw scores, so two searches that score on different scales can be combined fairly.",
    link: { href: "https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf", label: "The original paper (PDF)" },
  },
  ndcg: {
    term: "nDCG@10",
    kind: "how good an order is",
    says:
      "Looks only at the top ten results and rewards relevant ones more the higher they sit. 1.0 is the best order possible for that question; 0 means nothing relevant made the top ten. The score here is the average over every question.",
    here: `The search's own order scores ${floor.toFixed(3)}; the best re-ranker lifts it to ${jev.ndcg.toFixed(3)}.`,
    link: { href: "https://en.wikipedia.org/wiki/Discounted_cumulative_gain", label: "DCG and nDCG on Wikipedia" },
  },
  floor: {
    term: "The floor",
    kind: "what doing nothing gives you",
    says:
      "The order the search returned before any re-ranker touched it. Not re-ranking is a real option, so every re-ranker is measured against this order. Below the floor means a pipeline is better off without it.",
    here: `nDCG@10 ${floor.toFixed(3)}.`,
  },
  nfcorpus: {
    term: "NFCorpus",
    kind: "public search benchmark",
    says:
      "Health questions from NutritionFacts.org, PubMed abstracts that might answer them, and people's judgements of which abstracts answer which question. Part of BEIR, a standard collection of search benchmarks, so results are comparable with published work.",
    here: `${facts.queries} test questions, ${facts.documents?.toLocaleString()} abstracts.`,
    link: { href: "https://huggingface.co/datasets/BeIR/nfcorpus", label: "The dataset on Hugging Face" },
  },
  interval: {
    term: "95% interval",
    kind: "how sure the result is",
    says:
      "The range the true difference most likely sits in, given how much the result varies from question to question. When the whole range sits on one side of zero, the difference is unlikely to be luck.",
    link: { href: "https://en.wikipedia.org/wiki/Confidence_interval", label: "Confidence intervals on Wikipedia" },
  },
  p50: {
    term: "p50",
    kind: "typical time",
    says: "The median time of one call: half the calls were faster than this, half were slower.",
  },
} satisfies Record<string, GlossaryEntry>;

export type TermId = keyof typeof ENTRIES;
export const GLOSSARY: Record<TermId, GlossaryEntry> = ENTRIES;
