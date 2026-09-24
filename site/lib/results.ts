/** The results file, read as it is written by `rerank.evaluate`. */

export type Interval = [number, number] | null;

export interface Scores {
  distinct: number;
  of: number;
  min: number;
  max: number;
  stdev: number;
  tied: number;
  largest_group: number;
}

export interface Calls {
  n: number;
  retries: number;
  failed: number;
  input_tokens: number;
  market_cost_usd: number;
}

export interface MethodResult {
  method: string;
  "ndcg@10": number;
  "recall@10": number;
  "mrr@10": number;
  queries: number;
  "ndcg@10_ci95"?: Interval;
  "ndcg@10_vs_bm25"?: { mean: number; ci95: Interval; better: number; worse: number; same: number };
  latency: { n: number; p50_ms: number | null; p95_ms: number | null };
  scoring_wall_clock_s: number;
  calls?: Calls;
  scores?: Scores;
}

export interface Results {
  dataset: string;
  queries: number;
  top_k: number;
  device: string;
  commit: string | null;
  finished: string;
  methods: MethodResult[];
}

export interface TableRow {
  method: string;
  label: string;
  isFloor: boolean;
  ndcg: number;
  recall: number;
  mrr: number;
  delta: number | null;
  ci: Interval;
  excludesZero: boolean;
  better: number | null;
  worse: number | null;
  same: number | null;
  p50: number | null;
  p95: number | null;
  scores?: Scores;
  calls?: Calls;
}

const LABELS: Record<string, string> = {
  bm25: "BM25 (the floor)",
  "jev-score": "Jev score",
  "cross-encoder": "cross-encoder MiniLM-L-6",
  "laya-score": "Laya score",
  "laya-typed-score": "Laya typed-decisions score",
};

export function label(method: string): string {
  return LABELS[method] ?? method;
}

/**
 * Whether a 95% interval sits entirely on one side of zero. That is the whole
 * question a paired difference is asked: an interval that straddles zero says
 * the run could not tell the method from the floor, in either direction.
 */
export function excludesZero(ci: Interval | undefined): boolean {
  if (!ci) return false;
  const [low, high] = ci;
  return (low > 0 && high > 0) || (low < 0 && high < 0);
}

/** The share of a method's scored passages that sit in a tie with another. */
export function tieShare(scores: Scores | undefined): number {
  if (!scores || !scores.of) return 0;
  return scores.tied / scores.of;
}

/** One row per method, best nDCG@10 first. */
export function tableRows(results: Results): TableRow[] {
  return results.methods
    .map((m) => {
      const paired = m["ndcg@10_vs_bm25"];
      return {
        method: m.method,
        label: label(m.method),
        isFloor: m.method === "bm25",
        ndcg: m["ndcg@10"],
        recall: m["recall@10"],
        mrr: m["mrr@10"],
        delta: paired ? paired.mean : null,
        ci: paired ? paired.ci95 : null,
        excludesZero: excludesZero(paired?.ci95),
        better: paired ? paired.better : null,
        worse: paired ? paired.worse : null,
        same: paired ? paired.same : null,
        p50: m.latency.p50_ms,
        p95: m.latency.p95_ms,
        scores: m.scores,
        calls: m.calls,
      };
    })
    .sort((a, b) => b.ndcg - a.ndcg);
}

export function byMethod(results: Results, method: string): MethodResult | undefined {
  return results.methods.find((m) => m.method === method);
}

export const signed = (n: number, digits = 4) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
export const pct = (n: number, digits = 0) => `${(n * 100).toFixed(digits)}%`;

/**
 * A score range as a bar on a fixed axis, in percent. The two Laya checkpoints
 * answer on the same 0-to-4 scale, so laying their ranges on one axis is what
 * shows that the fine-tune never uses the bottom of it.
 */
export function spanBar(min: number, max: number, from: number, to: number): { left: number; width: number } {
  const size = to - from;
  const clamp = (n: number) => Math.min(100, Math.max(0, ((n - from) / size) * 100));
  const left = clamp(min);
  return { left, width: Math.max(clamp(max) - left, 0.5) };
}
