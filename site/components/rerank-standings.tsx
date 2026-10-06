import { floorRow, rangeAxis, rerankerRows, verdict, type Rerank, type Verdict } from "@/lib/rerank";

// Written out in full so Tailwind can find them: it reads class names, not code.
const TONE: Record<Verdict, { text: string; fill: string }> = {
  better: { text: "text-up", fill: "bg-up" },
  worse: { text: "text-down", fill: "bg-down" },
  "no clear change": { text: "text-ink-soft", fill: "bg-ink-soft" },
};

const ROW = "grid gap-x-8 gap-y-3 md:grid-cols-[minmax(14rem,1fr)_2fr] md:items-center";
const TRACK = "grid grid-cols-[1fr_7ch] items-center gap-4";

/**
 * Where each re-ranker left the ranking, on the absolute nDCG@10 scale.
 *
 * The interval chart further down answers "is it real"; this answers "how far",
 * in the unit the field reports. Every row starts at the floor's score, because
 * that is what the retriever already gave you, and runs to the method's own.
 * Drawn in HTML rather than SVG so a dot stays round at any width.
 */
export function RerankStandings({ data }: { data: Rerank }) {
  const floor = floorRow(data)!["ndcg@10"];
  const rows = rerankerRows(data);
  const scale = rangeAxis([floor, ...rows.map((r) => r.ndcg)], 100, 0.02);
  const at = (value: number) => `${scale.x(value)}%`;

  return (
    <figure className="mt-12 grid min-w-0 gap-4">
      <ol className="grid gap-px border-y border-line-strong bg-line">
        {rows.map((row) => {
          const said = verdict(row.mean, row.ci95);
          const tone = TONE[said];
          const from = Math.min(floor, row.ndcg);
          const to = Math.max(floor, row.ndcg);
          return (
            <li key={row.method} className={`${ROW} bg-page py-5`}>
              <div>
                <p className="text-lead font-semibold leading-tight">{row.label}</p>
                <p className="mt-1 text-micro text-ink-soft">
                  <span className={tone.text}>
                    {said === "no clear change" ? said : `${said} than doing nothing`}
                  </span>
                  <span className="numeric">
                    {" "}
                    · p50 {row.p50?.toFixed(1)} ms
                    {row.calls?.market_cost_usd ? ` · $${row.calls.market_cost_usd.toFixed(2)}` : " · local"}
                  </span>
                </p>
              </div>

              <div className={TRACK}>
                <div className="relative h-8" aria-hidden>
                  <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
                  <div className="absolute inset-y-1 w-px bg-ink-soft" style={{ left: at(floor) }} />
                  <div
                    className={`absolute top-1/2 h-1 -translate-y-1/2 ${tone.fill}`}
                    style={{ left: at(from), width: `${scale.x(to) - scale.x(from)}%` }}
                  />
                  <div
                    className={`absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-page ${tone.fill}`}
                    style={{ left: at(row.ndcg) }}
                  />
                </div>
                <p className="numeric text-right text-lead">
                  <span className="sr-only">
                    {row.label}: nDCG@10 {row.ndcg.toFixed(3)} against the floor&apos;s {floor.toFixed(3)}
                  </span>
                  <span aria-hidden>{row.ndcg.toFixed(3)}</span>
                </p>
              </div>
            </li>
          );
        })}
      </ol>

      <div className={ROW} aria-hidden>
        <div className="hidden md:block" />
        <div className={TRACK}>
          <div className="numeric relative h-5 text-micro text-ink-soft">
            {scale.ticks.map((t, i) => (
              <span
                key={t}
                className={`absolute top-0 ${
                  i === 0 ? "" : i === scale.ticks.length - 1 ? "-translate-x-full" : "-translate-x-1/2"
                }`}
                style={{ left: at(t) }}
              >
                {t.toFixed(2)}
              </span>
            ))}
          </div>
          <div />
        </div>
      </div>

      <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        nDCG@10 over all {data.summary.queries} questions. The grey stroke on every row is the retriever&apos;s
        own order, <span className="numeric">{floor.toFixed(3)}</span>; each bar runs from there to where the
        re-ranker left it. Times are the median per call, and the hosted model&apos;s is a network round trip,
        so it is a price rather than a speed.
      </figcaption>
    </figure>
  );
}
