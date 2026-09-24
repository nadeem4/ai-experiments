/**
 * The per-query CSV the page offers for download.
 *
 * It is written from the same file the strip plot is drawn from, so the download
 * and the chart cannot disagree, and nothing is rounded on the way out: the
 * numbers are the ones `pytrec_eval` produced. NFCorpus queries are free text and
 * some carry commas and quotation marks, so the fields are quoted per RFC 4180.
 */
import type { PerQuery } from "@/lib/per-query";

const NEEDS_QUOTES = /[",\n\r]/;

function field(value: string | number | boolean): string {
  const text = String(value);
  return NEEDS_QUOTES.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function perQueryCsv(data: PerQuery): string {
  const header = [
    "query_id",
    "query",
    "bm25_ndcg@10",
    ...data.methods,
    "judged_relevant_in_candidates",
    "has_example",
  ];
  const rows = data.rows.map((row) =>
    [
      row.id,
      row.query,
      row.bm25,
      ...data.methods.map((method) => row.deltas[method] ?? 0),
      row.relevant,
      row.example,
    ]
      .map(field)
      .join(","),
  );
  return [header.join(","), ...rows].join("\n");
}
