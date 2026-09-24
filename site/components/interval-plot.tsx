import { label, signed, tableRows, type Results } from "@/lib/results";

const WIDTH = 1000;
const LABEL = 20;
const BAR = 26;
const GAP = 14;
const AXIS = 30;
const ROW = LABEL + BAR;

/** A domain centred on zero, rounded out to the next hundredth. */
function half(values: number[]) {
  return Math.max(0.01, Math.ceil(Math.max(...values.map(Math.abs)) * 100 + 1) / 100);
}

/**
 * The four paired differences against BM25 with their 95% intervals.
 *
 * The table says every interval excludes zero. This draws it, because "excludes
 * zero" is a claim a reader should be able to check rather than accept: the zero
 * line is the same line in all four rows, and either a bar crosses it or it does
 * not. The intervals are against the common floor, so the picture says nothing
 * about how any two of these rows compare to each other.
 */
export function IntervalPlot({ results }: { results: Results }) {
  const rows = tableRows(results).filter((row) => row.ci);
  const h = half(rows.flatMap((row) => [row.ci![0], row.ci![1]]));
  const x = (value: number) => WIDTH / 2 + (value / h) * (WIDTH / 2);
  const height = rows.length * ROW + (rows.length - 1) * GAP + AXIS;
  const ticks = [-h, -h / 2, 0, h / 2, h];

  return (
    <figure className="grid min-w-0 gap-3">
      <div className="min-w-0 overflow-x-auto border border-line bg-surface p-4">
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="block h-auto w-full min-w-[520px]"
          role="img"
          aria-label={rows
            .map(
              (row) =>
                `${row.label}: ${signed(row.delta!)} nDCG@10, 95% interval ${signed(row.ci![0])} to ${signed(row.ci![1])}`,
            )
            .join(". ")}
        >
          {ticks
            .filter((t) => t !== 0)
            .map((t) => (
              <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={height - AXIS} stroke="var(--line)" />
            ))}
          <line
            x1={WIDTH / 2}
            x2={WIDTH / 2}
            y1={0}
            y2={height - AXIS}
            stroke="var(--line-strong)"
            strokeWidth={1.5}
          />

          {rows.map((row, i) => {
            const top = i * (ROW + GAP);
            const mid = top + LABEL + BAR / 2;
            const colour = row.delta! > 0 ? "var(--up)" : "var(--down)";
            return (
              <g key={row.method}>
                <text x={0} y={top + 13} className="numeric" fontSize={13} fill="var(--ink)">
                  {label(row.method)}
                </text>
                <text
                  x={WIDTH}
                  y={top + 13}
                  textAnchor="end"
                  className="numeric"
                  fontSize={13}
                  fill="var(--ink-soft)"
                >
                  {signed(row.delta!)} [{signed(row.ci![0])}, {signed(row.ci![1])}]
                </text>
                <line
                  x1={x(row.ci![0])}
                  x2={x(row.ci![1])}
                  y1={mid}
                  y2={mid}
                  stroke={colour}
                  strokeWidth={3}
                />
                {[row.ci![0], row.ci![1]].map((end) => (
                  <line
                    key={end}
                    x1={x(end)}
                    x2={x(end)}
                    y1={mid - 7}
                    y2={mid + 7}
                    stroke={colour}
                    strokeWidth={3}
                  />
                ))}
                <circle cx={x(row.delta!)} cy={mid} r={5} fill={colour} stroke="var(--surface)" strokeWidth={1.5} />
              </g>
            );
          })}

          <line
            x1={0}
            x2={WIDTH}
            y1={height - AXIS + 2}
            y2={height - AXIS + 2}
            stroke="var(--line-strong)"
          />
          {ticks.map((t) => (
            <text
              key={t}
              x={x(t)}
              y={height - AXIS + 20}
              textAnchor={t === ticks[0] ? "start" : t === ticks[ticks.length - 1] ? "end" : "middle"}
              className="numeric"
              fontSize={12}
              fill="var(--ink-soft)"
            >
              {t === 0 ? "0" : signed(t, 3)}
            </text>
          ))}
        </svg>
      </div>
    </figure>
  );
}
