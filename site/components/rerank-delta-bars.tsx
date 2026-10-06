"use client";

import { useMemo, useRef, useState } from "react";
import {
  barIndexAt,
  barLayout,
  barPaths,
  deltaSeries,
  signed,
  symmetricHalf,
  type Rerank,
  type RerankerRow,
} from "@/lib/rerank";

const WIDTH = 1000;
const HEIGHT = 120;
const STEP = 0.1;
const MIN_BAR = 0.8;

/**
 * Every query that could move, four times over, and the navigation for the
 * section above.
 *
 * A mean of +0.0478 and a mean of −0.0241 are two numbers of similar size; the
 * shapes behind them are not, and the shape is the finding. Sorting each method
 * by its own deltas turns a cloud into a profile: how much of the mass is above
 * the line, how long the tail below it runs, and how many queries the method
 * left exactly where it found them.
 *
 * Clicking a bar opens that query in the theatre above, under this method, so
 * the profile is also the way in.
 */
export function RerankDeltaBars({
  data,
  rows,
  selected,
  onSelect,
}: {
  data: Rerank;
  rows: RerankerRow[];
  selected: string;
  onSelect: (queryId: string, method: string) => void;
}) {
  // One scale for all four, so a taller bar always means a bigger change.
  const half = useMemo(
    () => symmetricHalf(rows.flatMap((row) => deltaSeries(data, row.method).map((p) => p.delta)), STEP),
    [data, rows],
  );
  return (
    <div className="grid min-w-0 gap-6">
      {rows.map((row) => (
        <Panel key={row.method} data={data} row={row} half={half} selected={selected} onSelect={onSelect} />
      ))}
    </div>
  );
}

function Panel({
  data,
  row,
  half,
  selected,
  onSelect,
}: {
  data: Rerank;
  row: RerankerRow;
  half: number;
  selected: string;
  onSelect: (queryId: string, method: string) => void;
}) {
  const layout = useMemo(
    () => barLayout(deltaSeries(data, row.method), { width: WIDTH, height: HEIGHT, step: STEP, gap: 1, half }),
    [data, row.method, half],
  );
  const paths = useMemo(() => barPaths(layout.bars, MIN_BAR), [layout]);

  // Counted off the bars that are drawn rather than taken from the summary: the
  // summary's "unchanged" is over all the queries and includes the ones that
  // could never move, which are not in this picture.
  const drawn = useMemo(
    () =>
      layout.bars.reduce(
        (tally, bar) => {
          if (bar.delta > 0) tally.better += 1;
          else if (bar.delta < 0) tally.worse += 1;
          else tally.same += 1;
          return tally;
        },
        { better: 0, worse: 0, same: 0 },
      ),
    [layout],
  );

  const svg = useRef<SVGSVGElement>(null);
  const chosen = layout.bars.findIndex((bar) => bar.id === selected);
  // Where the keyboard and the pointer currently are. One cursor, one tab stop:
  // 252 bars in each of four charts is not a tab order anybody can use, so the
  // chart is entered once and the arrow keys move inside it.
  const [cursor, setCursor] = useState(0);
  const at = Math.min(Math.max(cursor, 0), layout.bars.length - 1);
  const under = layout.bars[at];

  const fromPointer = (event: React.PointerEvent<SVGSVGElement> | React.MouseEvent<SVGSVGElement>) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box || !box.width) return -1;
    return barIndexAt(((event.clientX - box.left) / box.width) * WIDTH, WIDTH, layout.bars.length);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const by: Record<string, number | undefined> = { ArrowRight: 1, ArrowLeft: -1 };
    const last = layout.bars.length - 1;
    if (by[event.key] !== undefined) {
      event.preventDefault();
      setCursor(Math.min(last, Math.max(0, at + by[event.key]!)));
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setCursor(event.key === "Home" ? 0 : last);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(under.id, row.method);
    }
  };

  const marker = (bar: (typeof layout.bars)[number], stroke: string, dashed: boolean) => (
    <rect
      x={bar.x - 1.5}
      y={Math.min(bar.y, layout.zeroY) - 2}
      width={bar.width + 3}
      height={Math.max(bar.height, MIN_BAR) + 4}
      fill="none"
      stroke={stroke}
      strokeWidth={1.5}
      strokeDasharray={dashed ? "3 2" : undefined}
    />
  );

  return (
    <figure className="grid min-w-0 gap-2">
      <figcaption className="numeric flex flex-wrap items-baseline justify-between gap-x-4 text-micro">
        <span className="text-ink">
          {row.label} <span className="text-ink-soft">· axis ±{layout.half.toFixed(1)} nDCG@10, shared</span>
        </span>
        <span style={{ color: row.mean > 0 ? "var(--up)" : "var(--down)" }}>
          mean {signed(row.mean)} · {drawn.better} better · {drawn.worse} worse · {drawn.same} unchanged
        </span>
      </figcaption>

      {/* The tab stop is the frame rather than the drawing, so the focus outline
          the whole site uses is not clipped by the frame's own scrolling. */}
      <div
        className="min-w-0 overflow-x-auto border border-line bg-surface"
        tabIndex={0}
        role="group"
        aria-label={`${row.label}: each of the ${layout.bars.length} queries that could move, sorted by its nDCG@10 change against the ${data.floor} floor. Mean ${signed(row.mean)}, ${drawn.better} better, ${drawn.worse} worse, ${drawn.same} unchanged. Arrow keys move along the queries, enter opens the one under the cursor.`}
        onKeyDown={onKeyDown}
      >
        <svg
          ref={svg}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="block h-auto w-full min-w-[560px] cursor-pointer"
          aria-hidden="true"
          onPointerMove={(event) => {
            const index = fromPointer(event);
            if (index >= 0) setCursor(index);
          }}
          onClick={(event) => {
            const index = fromPointer(event);
            if (index >= 0) onSelect(layout.bars[index].id, row.method);
          }}
        >
          {paths.flat && <path d={paths.flat} fill="var(--line-strong)" />}
          {paths.loss && <path d={paths.loss} fill="var(--down)" />}
          {paths.gain && <path d={paths.gain} fill="var(--up)" />}
          <line
            x1={0}
            x2={WIDTH}
            y1={layout.zeroY}
            y2={layout.zeroY}
            stroke="var(--line-strong)"
            strokeWidth={1}
          />
          {under && marker(under, "var(--ink-soft)", true)}
          {chosen >= 0 && marker(layout.bars[chosen], "var(--ink)", false)}
        </svg>
      </div>

      <p className="numeric min-h-[1.5rem] text-micro text-ink-soft" aria-live="polite">
        {under && (
          <>
            <span className="text-ink">{under.text}</span> <span>{under.id}</span>{" "}
            <span style={{ color: under.delta > 0 ? "var(--up)" : under.delta < 0 ? "var(--down)" : undefined }}>
              {signed(under.delta)}
            </span>{" "}
            {under.id === selected ? "· open above" : "· click or press enter to open it above"}
          </>
        )}
      </p>
    </figure>
  );
}
