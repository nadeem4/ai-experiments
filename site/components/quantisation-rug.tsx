import { rugBuckets } from "@/lib/rug";
import { label, type Scores } from "@/lib/results";

const WIDTH = 1000;
const FULL_BINS = 500;
const ZOOM_BINS = 400;
/** The magnified window is this fraction of each method's own range, so the two
 *  are magnified by the same amount rather than by the same number of points. */
const ZOOM = 1 / 16;
const STRIP = 34;
const LABEL = 20;
const FOOT = 16;
const BLOCK = LABEL + STRIP + FOOT;
const GAP = 26;

export interface RugStrip {
  method: string;
  scores: Scores;
  values: number[];
}

function strip(values: number[], min: number, max: number, bins: number, top: number) {
  const counts = rugBuckets(values, min, max, bins);
  const busiest = Math.max(1, ...counts);
  const step = WIDTH / bins;
  return counts.map((count, bin) =>
    count === 0 ? null : (
      <rect
        key={bin}
        x={bin * step}
        y={top}
        width={Math.max(step - 0.5, 0.8)}
        height={STRIP}
        fill="var(--up)"
        opacity={0.3 + 0.7 * (count / busiest)}
      />
    ),
  );
}

/**
 * Every distinct value each method returned, laid on its own observed range.
 *
 * Jev's answers arrive through the gateway rounded to two decimals, so a 0-to-4
 * score has at most 401 places to land. The cross-encoder emits a raw logit and
 * has no such grid. That difference is the whole explanation for Jev's ties, and
 * a tie is not an opinion: tied passages keep the order BM25 gave them.
 *
 * A logit and a 0-to-4 score are not the same quantity and are not drawn as if
 * they were. Each strip is stretched over its own minimum and maximum, and the
 * magnified pair shows the same fraction of each range, so what is being
 * compared is resolution and nothing else.
 */
export function QuantisationRug({ strips }: { strips: RugStrip[] }) {
  const rows = strips.length * 2;
  const height = rows * BLOCK + (rows - 1) * GAP;

  return (
    <figure className="grid min-w-0 gap-3">
      <div className="min-w-0 overflow-x-auto border border-line bg-surface p-4">
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="block h-auto w-full min-w-[520px]"
          role="img"
          aria-label={strips
            .map(
              (s) =>
                `${label(s.method)} returned ${s.scores.distinct.toLocaleString()} distinct values across ${s.scores.of.toLocaleString()} scored calls, between ${s.scores.min} and ${s.scores.max}`,
            )
            .join(". ")}
        >
          {strips.flatMap((s, i) => {
            const span = s.scores.max - s.scores.min;
            const zoomFrom = s.scores.min + span * 0.5;
            const zoomTo = zoomFrom + span * ZOOM;
            return [
              { s, top: (2 * i) * (BLOCK + GAP), from: s.scores.min, to: s.scores.max, bins: FULL_BINS, zoomed: false },
              { s, top: (2 * i + 1) * (BLOCK + GAP), from: zoomFrom, to: zoomTo, bins: ZOOM_BINS, zoomed: true },
            ];
          }).map(({ s, top, from, to, bins, zoomed }) => (
            <g key={`${s.method}-${zoomed ? "zoom" : "full"}`}>
              <text x={0} y={top + 13} className="numeric" fontSize={13} fill="var(--ink)">
                {label(s.method)}
                {zoomed ? ", one sixteenth of that range" : ""}
              </text>
              <text
                x={WIDTH}
                y={top + 13}
                textAnchor="end"
                className="numeric"
                fontSize={13}
                fill="var(--ink-soft)"
              >
                {zoomed
                  ? `${rugBuckets(s.values, from, to, bins).filter(Boolean).length} values in view`
                  : `${s.scores.distinct.toLocaleString()} distinct of ${s.scores.of.toLocaleString()} calls`}
              </text>
              <rect x={0} y={top + LABEL} width={WIDTH} height={STRIP} fill="var(--surface-sunk)" />
              {strip(s.values, from, to, bins, top + LABEL)}
              <text x={0} y={top + LABEL + STRIP + 13} className="numeric" fontSize={12} fill="var(--ink-soft)">
                {from.toFixed(zoomed ? 3 : 2)}
              </text>
              <text
                x={WIDTH}
                y={top + LABEL + STRIP + 13}
                textAnchor="end"
                className="numeric"
                fontSize={12}
                fill="var(--ink-soft)"
              >
                {to.toFixed(zoomed ? 3 : 2)}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </figure>
  );
}
