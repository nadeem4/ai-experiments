/**
 * The page as a story: one question followed from the search to its grade,
 * then the whole run graded eight ways.
 *
 * The question is chosen, not picked by hand, and the Watch section further
 * down opens on the same one, so the example at the top and the instrument at
 * the bottom can never drift apart. Every place, score and verdict the story
 * states is read from the run here.
 */
import {
  label,
  openingQuery,
  pairFrom,
  queryNdcg,
  rerankerRows,
  verdict,
  type Detail,
  type Paired,
  type Rerank,
  type Verdict,
} from "@/lib/rerank";

// The run stores each change against the floor to four decimals, so a sum of
// the two is only good to four as well.
const four = (n: number) => Math.round(n * 1e4) / 1e4;

/** The re-rankers the question is chosen by: the best on average, and the worst. */
export function storyMethods(data: Rerank): { helped: string; hurt: string } {
  const rows = rerankerRows(data);
  return { helped: rows[0].method, hurt: rows[rows.length - 1].method };
}

/** The question the story follows and the Watch section opens on. */
export function storyQuestionId(data: Rerank): string {
  const { helped, hurt } = storyMethods(data);
  return openingQuery(data, helped, hurt);
}

export interface StoryReader {
  method: string;
  label: string;
  /** The 20, best first, as this re-ranker ordered them. */
  order: string[];
  /** Aligned to `order`. */
  scores: number[];
  /** Where it put the answer, counting from 1. */
  at: number;
  /** nDCG@10 of its order on this question. */
  ndcg: number;
}

export interface StoryQuestion {
  id: string;
  text: string;
  /** The relevant abstract the search ranked highest: the one the story follows. */
  answer: string;
  /** The search's 20, in its order. */
  results: { id: string; title: string }[];
  searchAt: number;
  searchNdcg: number;
  /** The best re-ranker, which the idea section shows reading the 20. */
  helped: StoryReader;
  /** Whichever re-ranker put the answer lowest. */
  buried: StoryReader;
}

/** Everything the first three parts need about one question, small enough to send to the browser. */
export function storyQuestion(data: Rerank, detail: Detail): StoryQuestion {
  const id = storyQuestionId(data);
  const query = data.queries.find((q) => q.id === id)!;
  const orders = detail.queries[id];
  const search = orders.orders[data.floor];
  const answer = [...query.relevant].sort((a, b) => search.indexOf(a) - search.indexOf(b))[0];

  const reader = (method: string): StoryReader => ({
    method,
    label: label(method),
    order: orders.orders[method],
    scores: orders.scores[method],
    at: orders.orders[method].indexOf(answer) + 1,
    ndcg: four(queryNdcg(query, method, data.floor).value ?? 0),
  });
  const readers = rerankerRows(data).map((row) => reader(row.method));
  const buried = readers.reduce((low, r) => (r.at > low.at ? r : low));

  return {
    id,
    text: query.text,
    answer,
    // Titles, not the passage text: a title is what a person scanning results reads.
    results: search.map((doc) => ({ id: doc, title: data.titles[doc] ?? detail.passages[doc] ?? doc })),
    searchAt: search.indexOf(answer) + 1,
    searchNdcg: four(queryNdcg(query, data.floor, data.floor).value ?? 0),
    helped: reader(storyMethods(data).helped),
    buried,
  };
}

const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
  "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
  "nineteenth", "twentieth"];

/** A place in words, as a sentence says it: "seventh", not "7th". */
export function ordinal(n: number): string {
  if (n >= 1 && n <= ORDINALS.length) return ORDINALS[n - 1];
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${suffix}`;
}

/** What a place in the top ten is worth to nDCG@10, and below it nothing. */
export function placeWorth(place: number): number {
  return place <= 10 ? 1 / Math.log2(place + 1) : 0;
}

export interface Measure {
  key: string;
  /** What a reader calls it. */
  label: string;
  /** What a paper calls it. */
  tech: string;
  /** The question the measure answers, in one line. */
  ask: string;
  /** Read as a share of questions, so a gap is in points rather than percent. */
  share?: boolean;
}

/** Every way the run is graded, the headline first. */
export const MEASURES: Measure[] = [
  { key: "ndcg@10", label: "The whole top ten", tech: "nDCG@10", ask: "How good is the order of the whole top ten? The headline measure." },
  { key: "success@1", label: "Top result right", tech: "success@1", ask: "Was the very first result a right answer?", share: true },
  { key: "mrr@10", label: "First right answer", tech: "MRR@10", ask: "How soon does the first right answer appear?" },
  { key: "ndcg@3", label: "The top three", tech: "nDCG@3", ask: "How good are just the first three places?" },
  { key: "ndcg@5", label: "The top five", tech: "nDCG@5", ask: "How good are the first five?" },
  { key: "recall@10", label: "Right answers found", tech: "Recall@10", ask: "Of all the right answers, what share made the top ten?" },
  { key: "p@10", label: "Share of the top ten right", tech: "P@10", ask: "Of the top ten, what share are right answers?" },
  { key: "map@10", label: "Average precision", tech: "MAP@10", ask: "How high the right answers sit, averaged over each one found." },
];

export interface BoardRow {
  method: string;
  label: string;
  value: number;
  paired: Paired;
  verdict: Verdict;
}

const valueOf = (data: Rerank, method: string, key: string) =>
  (data.summary.methods.find((m) => m.method === method) as unknown as Record<string, number> | undefined)?.[key];

/**
 * One measure: the floor's value, then each re-ranker's with its paired
 * difference and what that difference allows. The rows keep the headline order
 * whatever the measure, so switching measures moves bars rather than rows.
 */
export function measureBoard(data: Rerank, key: string): { floor: number; rows: BoardRow[] } | undefined {
  const comparisons = data.summary.by_metric?.[key];
  const floor = valueOf(data, data.floor, key);
  if (!comparisons || floor === undefined) return undefined;
  const rows = rerankerRows(data)
    .filter((row) => comparisons.vs_floor[row.method])
    .map((row) => {
      const paired = comparisons.vs_floor[row.method];
      return {
        method: row.method,
        label: row.label,
        value: valueOf(data, row.method, key) ?? 0,
        paired,
        verdict: verdict(paired.mean, paired.ci95),
      };
    });
  return { floor, rows };
}

/** `method` against `against` on one measure, question by question. */
export function pairOn(data: Rerank, key: string, method: string, against: string): Paired | undefined {
  const pairs = data.summary.by_metric?.[key]?.head_to_head;
  return pairs ? pairFrom(pairs, method, against) : undefined;
}
