/**
 * One line per passage, from the position BM25 gave it to the position the
 * re-ranker gave it.
 *
 * The lines are returned in BM25's order rather than the re-ranker's, because
 * BM25 is the fixed reference on this page and reading the chart means starting
 * from the left column. A passage the re-ranker never returned -- a call that
 * failed, so nothing was invented for it -- keeps its BM25 position, which is
 * exactly what the ranking code does with it.
 */

export interface SlopeLine {
  docId: string;
  /** 1-based position in BM25's ranking. */
  from: number;
  /** 1-based position after re-ranking. */
  to: number;
  /** Places moved up; negative is down. */
  moved: number;
  relevant: boolean;
  grade: number;
}

export function slopeLines(
  bm25: string[],
  order: string[],
  relevant: Record<string, number>,
): SlopeLine[] {
  if (!order.length) return [];
  const to = new Map(order.map((docId, i) => [docId, i + 1]));
  return bm25.map((docId, i) => {
    const from = i + 1;
    const landed = to.get(docId) ?? from;
    const grade = relevant[docId] ?? 0;
    return { docId, from, to: landed, moved: from - landed, relevant: grade > 0, grade };
  });
}
