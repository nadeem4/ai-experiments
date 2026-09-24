"use client";

import { useState } from "react";
import { perQueryCsv } from "@/lib/csv";
import type { PerQuery } from "@/lib/per-query";

/**
 * Everything the page was drawn from, in the form it was drawn from.
 *
 * The CSV is built in the browser from the same file the strip plot reads, so a
 * reader who checks the download against the chart is checking one set of
 * numbers, not two.
 */
export function Downloads({ perQuery, records }: { perQuery: PerQuery; records: number }) {
  const [failed, setFailed] = useState(false);

  const save = () => {
    try {
      const blob = new Blob([perQueryCsv(perQuery)], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `ndcg10-per-query-${perQuery.run}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setFailed(true);
    }
  };

  return (
    <ul className="grid max-w-[72ch] gap-3 text-body">
      <li className="border-l border-line pl-4">
        <a href="/results.json" download className="underline decoration-line-strong underline-offset-4 hover:decoration-ink">
          results.json
        </a>{" "}
        <span className="text-ink-soft">
          is the file every number on this page is read out of, written by the run itself.
        </span>
      </li>
      <li className="border-l border-line pl-4">
        <button
          type="button"
          onClick={save}
          className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          ndcg10-per-query.csv
        </button>{" "}
        <span className="text-ink-soft">
          is one row per query, {perQuery.queries} of them: BM25&apos;s own nDCG@10 and each method&apos;s
          difference from it. It is the strip plot as a spreadsheet.
        </span>
        {failed && (
          <span role="status" className="block text-micro text-down">
            This browser would not build the file. The same numbers are in{" "}
            <a
              href="https://github.com/nadeem4/ai-experiments/blob/main/site/data/per-query.json"
              className="underline"
            >
              per-query.json
            </a>
            .
          </span>
        )}
      </li>
      <li className="border-l border-line pl-4">
        <a
          href="https://github.com/nadeem4/ai-experiments/tree/main/site/public/examples"
          className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          The 24 example bundles
        </a>{" "}
        <span className="text-ink-soft">
          carry one query each: the candidates, every method&apos;s ordering, the official judgements, and the
          exact request and response for all 80 calls behind it. The run log all {records.toLocaleString()} of
          them came from is in the repository, not on this site.
        </span>
      </li>
    </ul>
  );
}
