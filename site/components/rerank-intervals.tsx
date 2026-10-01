import { axis, rerankerRows, signed, type Rerank } from "@/lib/rerank";

const WIDTH = 1000;
const LABEL = 20;
const BAND = 28;
const GAP = 12;
const AXIS = 32;
const ROW = LABEL + BAND;

/**
 * The whole result on one line.
 *
 * Four paired differences against the floor, each with the 95% interval it was
 * measured with, sharing one zero. Zero is the retriever's own order, so the
 * only question the chart is asked is which side of that line a method's whole
 * interval sits on, and that is a thing a reader can check rather than accept.
 *
 * The intervals are each against the common floor, so nothing here says how any
 * two of these four compare with one another.
 */
export function RerankIntervals({ data }: { data: Rerank }) {
  const rows = rerankerRows(data);
  const scale = axis(
    rows.flatMap((row) => [row.ci95?.[0] ?? row.mean, row.ci95?.[1] ?? row.mean]),
    WIDTH,
    0.01,
  );
  const height = rows.length * ROW + (rows.length - 1) * GAP + AXIS;

  return (
    <figure className="mt-10 grid min-w-0 gap-3">
      <div className="min-w-0 overflow-x-auto border border-line bg-surface p-4">
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="block h-auto w-full min-w-[540px]"
          role="img"
          aria-label={`nDCG@10 against the ${data.floor} floor, paired per query. ${rows
            .map(
              (row) =>
                `${row.method}: ${signed(row.mean)}, 95% interval ${signed(row.ci95![0])} to ${signed(row.ci95![1])}`,
            )
            .join(". ")}.`}
        >
          {scale.ticks
            .filter((t) => t !== 0)
            .map((t) => (
              <line key={t} x1={scale.x(t)} x2={scale.x(t)} y1={0} y2={height - AXIS} stroke="var(--line)" />
            ))}
          <line
            x1={scale.x(0)}
            x2={scale.x(0)}
            y1={0}
            y2={height - AXIS}
            stroke="var(--line-strong)"
            strokeWidth={1.5}
          />

          {rows.map((row, i) => {
            const top = i * (ROW + GAP);
            const mid = top + LABEL + BAND / 2;
            const colour = row.mean > 0 ? "var(--up)" : "var(--down)";
            const [low, high] = row.ci95 ?? [row.mean, row.mean];
            return (
              <g key={row.method}>
                <text x={0} y={top + 13} className="numeric" fontSize={13} fill="var(--ink)">
                  {row.method}
                </text>
                <text
                  x={WIDTH}
                  y={top + 13}
                  textAnchor="end"
                  className="numeric"
                  fontSize={13}
                  fill="var(--ink-soft)"
                >
                  {signed(row.mean)} [{signed(low)}, {signed(high)}]
                </text>
                <line x1={scale.x(low)} x2={scale.x(high)} y1={mid} y2={mid} stroke={colour} strokeWidth={3} />
                {[low, high].map((end) => (
                  <line
                    key={end}
                    x1={scale.x(end)}
                    x2={scale.x(end)}
                    y1={mid - 8}
                    y2={mid + 8}
                    stroke={colour}
                    strokeWidth={3}
                  />
                ))}
                <circle
                  cx={scale.x(row.mean)}
                  cy={mid}
                  r={5}
                  fill={colour}
                  stroke="var(--surface)"
                  strokeWidth={1.5}
                />
              </g>
            );
          })}

          <line x1={0} x2={WIDTH} y1={height - AXIS + 2} y2={height - AXIS + 2} stroke="var(--line-strong)" />
          {scale.ticks.map((t) => (
            <text
              key={t}
              x={scale.x(t)}
              y={height - AXIS + 20}
              textAnchor={t === scale.ticks[0] ? "start" : t === scale.ticks[scale.ticks.length - 1] ? "end" : "middle"}
              className="numeric"
              fontSize={12}
              fill="var(--ink-soft)"
            >
              {t === 0 ? `0 — the ${data.floor} floor` : signed(t, 3)}
            </text>
          ))}
        </svg>
      </div>
      <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        nDCG@10 against the <span className="numeric">{data.floor}</span> floor, paired per query, with 95%
        intervals. Zero is the order the retriever already gave you. Each interval is against that common
        floor, so the picture says nothing about how any two of these four compare with each other.
      </figcaption>
    </figure>
  );
}
