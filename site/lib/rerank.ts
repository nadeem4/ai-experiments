/**
 * Reading the re-ranking run.
 *
 * Every number the page shows comes through here, and the one rule the module
 * exists to hold is that `hybrid` is the floor rather than a fifth contender:
 * it is what the retriever already gave you, so it has no delta, no interval and
 * no scores of its own, and a re-ranker's result is only ever the distance from
 * it. Anything that cannot be read out of the run comes back as `null` so the
 * page can say it is missing instead of drawing the floor's answer under another
 * model's name.
 */

/** A 95% interval as the run writes it: two numbers, low then high. */
export type Interval = number[] | null;

export interface RerankScores {
  distinct: number;
  of: number;
  min: number;
  max: number;
  stdev: number;
  tied: number;
  largest_group: number;
}

export interface RerankCalls {
  n: number;
  retries: number;
  failed: number;
  input_tokens: number;
  market_cost_usd: number;
}

export interface Paired {
  mean: number;
  ci95: Interval;
  better: number;
  worse: number;
  same: number;
}

export interface RerankMethod {
  method: string;
  "ndcg@10": number;
  "recall@10": number;
  "mrr@10": number;
  queries: number;
  "ndcg@10_ci95"?: Interval;
  "ndcg@10_vs_floor"?: Paired;
  latency: { n: number; p50_ms: number | null; p95_ms: number | null };
  scoring_wall_clock_s: number;
  calls?: RerankCalls;
  scores?: RerankScores;
}

/** One pair as the run writes it: `method - against`. */
export type PairRow = Paired & { method: string; against: string };

export interface RerankQuery {
  id: string;
  text: string;
  relevant: string[];
  can_move: boolean;
  /** The floor's own nDCG@10 under `floor_ndcg@10`, every other key a delta. */
  ndcg: Record<string, number>;
}

export interface Rerank {
  floor: string;
  queries: RerankQuery[];
  titles: Record<string, string>;
  summary: {
    dataset: string;
    queries: number;
    top_k: number;
    device: string;
    commit: string | null;
    finished: string;
    /** The corpus every question was searched over. Absent from runs reported before it was recorded. */
    documents?: number;
    /** Every pair of re-rankers, `method - against`, per query. */
    head_to_head?: PairRow[];
    /** The same comparisons for every measure the run reports, nDCG@10 among them. */
    by_metric?: Record<string, { vs_floor: Record<string, Paired>; head_to_head: PairRow[] }>;
    methods: RerankMethod[];
  };
  config: Record<string, unknown>;
}

/** The 1.4MB companion, fetched rather than imported. */
export interface DetailQuery {
  orders: Record<string, string[]>;
  /** Aligned to the order of the same name, position by position. */
  scores: Record<string, number[]>;
}

export interface Detail {
  queries: Record<string, DetailQuery>;
  passages: Record<string, string>;
}

const FLOOR_NDCG = "floor_ndcg@10";

const LABELS: Record<string, string> = {
  hybrid: "hybrid, the floor",
  "jev-score": "Jev",
  "cross-encoder": "Cross-encoder",
  "laya-score": "Laya",
  "laya-typed-score": "Laya typed",
};

export function label(method: string): string {
  return LABELS[method] ?? method;
}

export const signed = (n: number, digits = 4) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;

export interface RerankerRow {
  method: string;
  label: string;
  ndcg: number;
  recall: number;
  mrr: number;
  mean: number;
  ci95: Interval;
  excludesZero: boolean;
  better: number;
  worse: number;
  same: number;
  p50: number | null;
  p95: number | null;
  calls?: RerankCalls;
  scores?: RerankScores;
}

/** Whether a 95% interval stays on one side of zero, which is the only thing a
 * paired difference is asked. */
export function excludesZero(ci: Interval | undefined): boolean {
  if (!ci) return false;
  const [low, high] = ci;
  return (low > 0 && high > 0) || (low < 0 && high < 0);
}

/** The methods that re-ranked something, best first. The floor is not among
 * them: it is the line they are measured from. */
export function rerankerRows(data: Rerank): RerankerRow[] {
  return data.summary.methods
    .filter((m) => m.method !== data.floor && m["ndcg@10_vs_floor"])
    .map((m) => {
      const paired = m["ndcg@10_vs_floor"]!;
      return {
        method: m.method,
        label: label(m.method),
        ndcg: m["ndcg@10"],
        recall: m["recall@10"],
        mrr: m["mrr@10"],
        mean: paired.mean,
        ci95: paired.ci95,
        excludesZero: excludesZero(paired.ci95),
        better: paired.better,
        worse: paired.worse,
        same: paired.same,
        p50: m.latency.p50_ms,
        p95: m.latency.p95_ms,
        calls: m.calls,
        scores: m.scores,
      };
    })
    .sort((a, b) => b.mean - a.mean);
}

export function floorRow(data: Rerank): RerankMethod | undefined {
  return data.summary.methods.find((m) => m.method === data.floor);
}

/** The floor, then the re-rankers best first: the order the page reads them in. */
export function methodOrder(data: Rerank): string[] {
  return [data.floor, ...rerankerRows(data).map((r) => r.method)];
}

export interface RunFacts {
  calls: number;
  failed: number;
  spendUsd: number;
  queries: number;
  topK: number;
  device: string;
  dataset: string;
  passages: number;
  documents: number | null;
}

/** The size of the run, added up rather than quoted. */
export function runFacts(data: Rerank): RunFacts {
  const methods = data.summary.methods;
  return {
    calls: methods.reduce((total, m) => total + (m.calls?.n ?? 0), 0),
    failed: methods.reduce((total, m) => total + (m.calls?.failed ?? 0), 0),
    spendUsd: methods.reduce((total, m) => total + (m.calls?.market_cost_usd ?? 0), 0),
    queries: data.summary.queries,
    topK: data.summary.top_k,
    device: data.summary.device,
    dataset: data.summary.dataset,
    passages: Object.keys(data.titles).length,
    documents: data.summary.documents ?? null,
  };
}

/** `method` against `against` out of a list that holds each pair once, either way round. */
export function pairFrom(pairs: PairRow[], method: string, against: string): Paired | undefined {
  const same = pairs.find((p) => p.method === method && p.against === against);
  if (same) return { mean: same.mean, ci95: same.ci95, better: same.better, worse: same.worse, same: same.same };
  const turned = pairs.find((p) => p.method === against && p.against === method);
  if (!turned) return undefined;
  return {
    mean: -turned.mean,
    ci95: turned.ci95 ? [-turned.ci95[1], -turned.ci95[0]] : null,
    better: turned.worse,
    worse: turned.better,
    same: turned.same,
  };
}

export type Verdict = "better" | "worse" | "no clear change";

/** What a paired difference allows the page to say, and nothing more. */
export function verdict(mean: number, ci: Interval | undefined): Verdict {
  if (!excludesZero(ci)) return "no clear change";
  return mean > 0 ? "better" : "worse";
}

/** How many queries had something relevant to find, and how many did not. */
export function movableSplit(data: Rerank): { movable: number; excluded: number } {
  const movable = data.queries.filter((q) => q.can_move).length;
  return { movable, excluded: data.queries.length - movable };
}

/** What this query scored under this method, and what the floor scored on it.
 * The run records the floor's value and each method's distance from it, so the
 * method's own value is the sum; a method with no recorded distance gets none. */
export function queryNdcg(
  query: RerankQuery,
  method: string,
  floor: string,
): { floor: number; value: number | null; delta: number | null } {
  const at = query.ndcg[FLOOR_NDCG] ?? 0;
  if (method === floor) return { floor: at, value: at, delta: 0 };
  const delta = query.ndcg[method];
  if (delta === undefined) return { floor: at, value: null, delta: null };
  return { floor: at, value: at + delta, delta };
}

/**
 * The question the explorer opens on: the one where `helped` raised nDCG@10 and
 * `hurt` lowered it by the widest combined margin. Opening on the best re-ranker's
 * single best question showed every method helping, which is the opposite of
 * the finding; a question where two of them disagree shows what the means hide.
 */
export function openingQuery(data: Rerank, helped: string, hurt: string): string {
  const split = data.queries.filter((q) => q.can_move && q.ndcg[helped] > 0 && q.ndcg[hurt] < 0);
  if (!split.length) return (data.queries.find((q) => q.can_move) ?? data.queries[0]).id;
  const margin = (q: RerankQuery) => q.ndcg[helped] - q.ndcg[hurt];
  return split.reduce((a, b) => (margin(b) > margin(a) ? b : a)).id;
}

/**
 * The query list, filtered by what someone typed into the search box.
 *
 * `keep` is the question currently on screen, and it stays in the list even
 * when the search would drop it: it can be chosen from the chart further down
 * the page while a search is still narrowing the list, and a picker showing
 * neither the current question nor any sign of it is worse than one showing it
 * out of order.
 */
export function matchQueries(queries: RerankQuery[], term: string, keep?: string): RerankQuery[] {
  const needle = term.trim().toLowerCase();
  const found = needle
    ? queries.filter((q) => q.text.toLowerCase().includes(needle) || q.id.toLowerCase().includes(needle))
    : queries;
  if (!keep || found.some((q) => q.id === keep)) return found;
  const current = queries.find((q) => q.id === keep);
  return current ? [current, ...found] : found;
}
