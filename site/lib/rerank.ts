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
  "cross-encoder": "cross-encoder MiniLM",
  "laya-score": "Laya",
  "laya-typed-score": "Laya typed-decisions",
};

export function label(method: string): string {
  return LABELS[method] ?? method;
}

export const signed = (n: number, digits = 4) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
export const pct = (n: number, digits = 0) => `${(n * 100).toFixed(digits)}%`;

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
  };
}

/** Half-width of a domain centred on zero that covers every value, rounded out
 * to the next `step` so the axis lands on ticks a reader can name. */
export function symmetricHalf(values: number[], step: number): number {
  const extent = Math.max(0, ...values.map(Math.abs));
  return Math.max(step, Math.ceil(extent / step - 1e-9) * step);
}

export interface Axis {
  half: number;
  x: (value: number) => number;
  ticks: number[];
}

/** A horizontal scale centred on zero. Zero is the whole point of these charts,
 * so it sits at the middle of the drawing rather than wherever the data ends up
 * putting it. */
export function axis(values: number[], width: number, step: number): Axis {
  const half = symmetricHalf(values, step);
  return {
    half,
    x: (value: number) => width / 2 + (value / half) * (width / 2),
    ticks: [-half, -half / 2, 0, half / 2, half],
  };
}

export interface DeltaPoint {
  id: string;
  text: string;
  delta: number;
}

/** One method's change against the floor, per query, largest gain first.
 * Only the queries that could move: for the rest every method scores zero
 * whatever order it picks, so a bar of zero would say nothing. */
export function deltaSeries(data: Rerank, method: string): DeltaPoint[] {
  return data.queries
    .filter((q) => q.can_move && q.ndcg[method] !== undefined)
    .map((q) => ({ id: q.id, text: q.text, delta: q.ndcg[method] }))
    .sort((a, b) => b.delta - a.delta);
}

/** How many queries had something relevant to find, and how many did not. */
export function movableSplit(data: Rerank): { movable: number; excluded: number } {
  const movable = data.queries.filter((q) => q.can_move).length;
  return { movable, excluded: data.queries.length - movable };
}

export interface Bar {
  id: string;
  text: string;
  delta: number;
  x: number;
  width: number;
  y: number;
  height: number;
}

export interface BarLayout {
  bars: Bar[];
  zeroY: number;
  half: number;
}

/** Every movable query as one thin bar, hung off a zero line in the middle. */
export function barLayout(
  series: DeltaPoint[],
  { width, height, step, gap = 0 }: { width: number; height: number; step: number; gap?: number },
): BarLayout {
  const half = symmetricHalf(
    series.map((point) => point.delta),
    step,
  );
  const zeroY = height / 2;
  const slot = width / Math.max(1, series.length);
  const barWidth = Math.max(slot - gap, slot * 0.5);

  return {
    half,
    zeroY,
    bars: series.map((point, i) => {
      const length = (Math.abs(point.delta) / half) * (height / 2);
      return {
        ...point,
        x: i * slot,
        width: barWidth,
        y: point.delta >= 0 ? zeroY - length : zeroY,
        height: length,
      };
    }),
  };
}

/**
 * The bars as three paths rather than 252 elements.
 *
 * A bar per query, four methods deep, is a thousand shapes, and a thousand
 * shapes is a thousand tags in the HTML and a thousand tab stops in the page.
 * Drawn as one path per direction the chart costs a few hundred bytes, and the
 * one thing a path cannot do -- be pointed at -- is done by arithmetic in
 * `barIndexAt` instead.
 */
export function barPaths(bars: Bar[], minHeight = 0.8): { gain: string; loss: string; flat: string } {
  const round = (n: number) => Math.round(n * 100) / 100;
  const out = { gain: "", loss: "", flat: "" };
  for (const bar of bars) {
    const height = Math.max(bar.height, minHeight);
    const key = bar.delta > 0 ? "gain" : bar.delta < 0 ? "loss" : "flat";
    out[key] += `M${round(bar.x)} ${round(bar.y)}h${round(bar.width)}v${round(height)}h${-round(bar.width)}z`;
  }
  return out;
}

/** Which bar a pointer at `x` is over, in the drawing's own coordinates. */
export function barIndexAt(x: number, width: number, count: number): number {
  if (count <= 0) return -1;
  const index = Math.floor((x / width) * count);
  return Math.min(count - 1, Math.max(0, index));
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

export interface CandidateRow {
  rank: number;
  docId: string;
  title: string;
  score: number | null;
  relevant: boolean;
  floorRank: number;
  /** Places gained against the floor: positive travelled up the list. */
  moved: number;
}

/**
 * The candidate list as one method left it. Returns `null` when the run holds no
 * order for that method on that query, which is not the same as the method
 * having agreed with the floor and must not be drawn as though it were.
 */
export function candidateRows(
  entry: DetailQuery | undefined,
  method: string,
  floor: string,
  relevant: string[],
  titles: Record<string, string>,
): CandidateRow[] | null {
  const order = entry?.orders[method];
  if (!entry || !order) return null;

  const scores = entry.scores[method];
  const floorOrder = entry.orders[floor] ?? order;
  const judged = new Set(relevant);

  return order.map((docId, i) => {
    const floorRank = floorOrder.indexOf(docId) + 1;
    return {
      rank: i + 1,
      docId,
      title: titles[docId] ?? docId,
      score: scores?.[i] ?? null,
      relevant: judged.has(docId),
      floorRank,
      moved: floorRank ? floorRank - (i + 1) : 0,
    };
  });
}

/**
 * Where to scroll the query list so its chosen row is visible, moving as little
 * as possible. `scrollIntoView` does the same arithmetic but applies it to every
 * scrollable ancestor too, the window included, which is how the page used to
 * open halfway down at the picker instead of at its title.
 */
export function scrollTopToShow(
  row: { top: number; height: number },
  view: { scrollTop: number; height: number },
): number {
  if (row.top < view.scrollTop) return row.top;
  if (row.top + row.height > view.scrollTop + view.height) return row.top + row.height - view.height;
  return view.scrollTop;
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

/**
 * What to say about one passage's number. A missing score means two different
 * things and the page must not blur them: the floor never scored anything
 * because it is the retriever's own order, whereas a scorer with no number
 * there is a call that failed. The run kept that passage where the retriever
 * had put it, so the rank beside it is the first stage's, not the scorer's.
 */
export function scoreNote(method: string, floor: string, score: number | null): string {
  if (score !== null) return `${method} scored this passage ${score.toFixed(3)}`;
  if (method === floor)
    return `${floor} is the order the retriever returned, so it holds no score of its own.`;
  return `${method} returned nothing for this passage: the call failed, so it kept the slot ${floor} gave it.`;
}
