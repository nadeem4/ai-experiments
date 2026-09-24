/**
 * One row per query in the run, as `scripts/export_examples.py` writes it.
 *
 * The table and the strip plot read the same file, so they cannot drift apart:
 * the mean of a column here is the mean the results file reports. `example` is
 * the only site-shaped field, and it says whether that query's full wire was
 * exported -- 24 of the 323 were, because the wire is 25,840 records.
 */

export interface PerQueryRow {
  id: string;
  query: string;
  /** BM25's own nDCG@10 for this query, the floor the differences are against. */
  bm25: number;
  /** Each method's nDCG@10 minus BM25's, on this query's own candidates. */
  deltas: Record<string, number>;
  /** How many of the candidates the official qrels judge relevant. */
  relevant: number;
  example: boolean;
}

export interface PerQuery {
  run: string;
  queries: number;
  methods: string[];
  rows: PerQueryRow[];
}
