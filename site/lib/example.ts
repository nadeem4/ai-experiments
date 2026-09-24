/**
 * One exported query, and the two columns the page compares.
 *
 * BM25 is always the left-hand column, because it is the floor every method was
 * measured against; the picker only changes what sits beside it. Five columns of
 * twenty passages would be unreadable, and the comparison that matters is always
 * against the floor anyway.
 */

export interface Passage {
  title: string;
  snippet: string;
  chars: number;
}

export interface MethodOrdering {
  order: string[];
  scores: Record<string, number | null>;
  delta: number;
  failed: string[];
}

export interface WireRecord {
  request?: unknown;
  response?: unknown;
  latency_ms?: number;
  retries?: number;
  failed?: boolean;
}

export interface Example {
  query_id: string;
  query: string;
  reason: string;
  bm25: string[];
  relevant: Record<string, number>;
  passages: Record<string, Passage>;
  methods: Record<string, MethodOrdering>;
  wire: Record<string, Record<string, WireRecord>>;
}

export interface IndexEntry {
  query_id: string;
  query: string;
  reason: string;
  jev_delta: number;
  laya_delta: number;
  jev_failed: number;
  relevant: number;
  candidates: number;
}

export interface ExamplesIndex {
  run: string;
  queries_in_run: number;
  records_in_run: number;
  exported: IndexEntry[];
}

export interface Line {
  docId: string;
  position: number;
  /** Places moved up against BM25; negative is down, 0 is unchanged. */
  move: number;
  relevant: boolean;
  grade: number;
  score: number | null;
  failed: boolean;
  passage: Passage;
}

const REASONS: Record<string, string> = {
  "jev-gain": "Jev gained the most here",
  "laya-loss": "Laya lost the most here",
  unchanged: "nothing moved against BM25",
  "jev-failure": "a Jev call failed and the passage held its BM25 slot",
};

export function reasonLabel(reason: string): string {
  return REASONS[reason] ?? reason;
}

function line(ex: Example, docId: string, position: number, move: number, ordering?: MethodOrdering): Line {
  return {
    docId,
    position,
    move,
    relevant: (ex.relevant[docId] ?? 0) > 0,
    grade: ex.relevant[docId] ?? 0,
    score: ordering ? ordering.scores[docId] ?? null : null,
    failed: ordering ? ordering.failed.includes(docId) : false,
    passage: ex.passages[docId],
  };
}

/** BM25's fixed ordering, and one method's re-ordering of the same candidates. */
export function columns(ex: Example, method: string): { bm25: Line[]; method: Line[] } {
  const floor = new Map(ex.bm25.map((doc, i) => [doc, i]));
  const ordering = ex.methods[method];
  return {
    bm25: ex.bm25.map((doc, i) => line(ex, doc, i + 1, 0)),
    method: (ordering?.order ?? []).map((doc, i) => line(ex, doc, i + 1, (floor.get(doc) ?? i) - i, ordering)),
  };
}
