import { Chip } from "@/components/chip";
import { byMethod, signed, tableRows, type Results } from "@/lib/results";

/**
 * The finding over every query in the run. Each interval is the paired per-query
 * difference against BM25, so a row is a result only when its interval sits clear
 * of zero, which is marked rather than left to the reader.
 */
export function ResultsTable({ results }: { results: Results }) {
  const rows = tableRows(results);
  const lead = byMethod(results, "jev-score")!["ndcg@10"] - byMethod(results, "cross-encoder")!["ndcg@10"];
  return (
    <figure className="grid min-w-0 gap-3">
      <div className="min-w-0 overflow-x-auto border border-line bg-surface">
        <table className="numeric w-full min-w-[720px] border-collapse text-left text-micro">
          <caption className="sr-only">
            nDCG@10, Recall@10 and MRR@10 for each method, with the paired per-query difference against BM25
          </caption>
          <thead>
            <tr className="border-b border-line-strong">
              <th scope="col" className="px-3 py-2 font-medium">Ranking</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">nDCG@10</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Recall@10</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">MRR@10</th>
              <th scope="col" className="px-3 py-2 font-medium">vs BM25, paired (95% CI)</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">p50</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">p95</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.method}
                className={`border-b border-line align-top last:border-0 ${row.isFloor ? "bg-sunk" : ""}`}
              >
                <th scope="row" className="px-3 py-2 text-left font-medium">{row.label}</th>
                <td className="px-3 py-2 text-right font-medium">{row.ndcg.toFixed(4)}</td>
                <td className="px-3 py-2 text-right text-ink-soft">{row.recall.toFixed(4)}</td>
                <td className="px-3 py-2 text-right text-ink-soft">{row.mrr.toFixed(4)}</td>
                <td className="px-3 py-2">
                  {row.delta === null ? (
                    <span className="text-ink-soft">the floor</span>
                  ) : (
                    <>
                      <b style={{ color: row.delta > 0 ? "var(--up)" : "var(--down)" }}>
                        {signed(row.delta)}
                      </b>
                      {row.ci && (
                        <span className="text-ink-soft">
                          {" "}
                          [{signed(row.ci[0])}, {signed(row.ci[1])}]
                        </span>
                      )}
                      <Chip>{row.excludesZero ? "excludes 0" : "crosses 0"}</Chip>
                      <span className="block text-ink-soft">
                        better on {row.better}, worse on {row.worse}, unchanged on {row.same}
                      </span>
                    </>
                  )}
                </td>
                <td className="px-3 py-2 text-right text-ink-soft">
                  {row.p50 === null ? "—" : `${row.p50} ms`}
                </td>
                <td className="px-3 py-2 text-right text-ink-soft">
                  {row.p95 === null ? "—" : `${row.p95} ms`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        All {results.queries} {results.dataset} queries, top {results.top_k} candidates each, official qrels.
        Every method re-ranks the same candidate list, so the difference is paired per query and BM25 is the
        floor. <b className="text-ink">Every interval in this run excludes zero, in both directions.</b> Jev is{" "}
        {signed(lead)} ahead of the cross-encoder, but that pair was never compared directly, and two intervals
        against a common floor do not settle it. Latencies are not comparable across rows: Jev is a network
        round trip, the local models ran on {results.device.toUpperCase()}.
      </figcaption>
    </figure>
  );
}
