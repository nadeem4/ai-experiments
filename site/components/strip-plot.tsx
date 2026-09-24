"use client";

import { useMemo } from "react";
import { stripLayout, type StripRow } from "@/lib/strip";
import type { PerQuery } from "@/lib/per-query";
import { label } from "@/lib/results";

const WIDTH = 1000;
/** The plot is inset from the frame so a dot at either extreme is a whole dot. */
const PAD = 10;
const PLOT = WIDTH - 2 * PAD;
const COLUMN = 3;
const BAND = 78;
const PITCH = 4;
const LABEL = 22;
const GAP = 34;
const AXIS = 34;
const ROW = LABEL + BAND;

const TICKS = [-1, -0.5, 0, 0.5, 1];

function tick(value: number) {
  return value === 0 ? "0" : `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(1)}`;
}

function signed(value: number, digits = 4) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

/**
 * All 323 queries, four times over: one dot per query, placed by how far that
 * method moved its nDCG@10 against BM25.
 *
 * The table reports four means with four intervals, and every one of them hides
 * the same thing: the largest single group of queries is the group where nothing
 * moved at all. Here that group is a column of dots on the zero line, as tall as
 * it is numerous, and the asymmetry of the four rows is the finding in one look.
 *
 * The chart is also the navigation. 24 of the 323 queries have their full wire
 * exported, and in the row of the method currently on screen those queries are
 * ringed and can be opened; every other dot is a reading, not a link.
 */
export function StripPlot({
  data,
  methods,
  active,
  selected,
  onSelect,
  means,
}: {
  data: PerQuery;
  methods: string[];
  active: string;
  selected: string;
  onSelect: (queryId: string) => void;
  means: Record<string, number>;
}) {
  const layout = useMemo(() => {
    const rows: StripRow[] = methods.map((method) => ({
      method,
      points: data.rows.map((row) => ({
        id: row.id,
        value: row.deltas[method] ?? 0,
        example: row.example,
      })),
    }));
    return stripLayout(rows, { width: PLOT, column: COLUMN, band: BAND, pitch: PITCH });
  }, [data, methods]);

  const height = methods.length * ROW + (methods.length - 1) * GAP + AXIS;
  const centreOf = (i: number) => i * (ROW + GAP) + LABEL + BAND / 2;
  const x = (value: number) => PAD + PLOT / 2 + (value / layout.half) * (PLOT / 2);

  return (
    <figure className="grid min-w-0 gap-3">
      <div className="min-w-0 overflow-x-auto border border-line bg-surface">
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="block h-auto w-full min-w-[620px]"
          role="img"
          aria-label={`Each of the ${data.queries} queries as one dot, in four rows, placed by its nDCG@10 change against BM25. Jev's dots sit mostly right of zero, both Laya rows sit mostly left, and every row has a tall pile of queries exactly on zero.`}
        >
          {methods.map((method, i) =>
            method === active ? (
              <rect
                key={method}
                x={0}
                y={i * (ROW + GAP) + LABEL - 4}
                width={WIDTH}
                height={BAND + 8}
                fill="var(--surface-sunk)"
              />
            ) : null,
          )}
          {TICKS.filter((t) => t !== 0).map((t) => (
            <line
              key={t}
              x1={x(t)}
              x2={x(t)}
              y1={0}
              y2={height - AXIS}
              stroke="var(--line)"
              strokeWidth={1}
            />
          ))}
          <line
            x1={x(0)}
            x2={x(0)}
            y1={0}
            y2={height - AXIS}
            stroke="var(--line-strong)"
            strokeWidth={1.5}
          />

          {layout.rows.map((row, i) => {
            const isActive = row.method === active;
            const mean = means[row.method] ?? 0;
            return (
              <g key={row.method}>
                <text
                  x={0}
                  y={i * (ROW + GAP) + 14}
                  className="numeric"
                  fontSize={13}
                  fontWeight={isActive ? 500 : 400}
                  fill={isActive ? "var(--ink)" : "var(--ink-soft)"}
                >
                  {label(row.method)}
                  {isActive ? "  ◂ shown below" : ""}
                </text>
                <text
                  x={WIDTH}
                  y={i * (ROW + GAP) + 14}
                  textAnchor="end"
                  className="numeric"
                  fontSize={13}
                  fill={mean > 0 ? "var(--up)" : "var(--down)"}
                >
                  mean {signed(mean)}
                </text>
                <line
                  x1={0}
                  x2={WIDTH}
                  y1={centreOf(i)}
                  y2={centreOf(i)}
                  stroke="var(--line)"
                  strokeWidth={1}
                />

                {row.points.map((point) => {
                  const cy = centreOf(i) + point.y;
                  const fill =
                    point.value > 0
                      ? "var(--up)"
                      : point.value < 0
                        ? "var(--down)"
                        : "var(--line-strong)";
                  if (!point.example) {
                    return (
                      <circle
                        key={point.id}
                        cx={PAD + point.x}
                        cy={cy}
                        r={2.4}
                        fill={fill}
                        opacity={0.55}
                      />
                    );
                  }
                  const open = point.id === selected;
                  const dot = (
                    <>
                      <circle
                        className="focus-ring"
                        cx={PAD + point.x}
                        cy={cy}
                        r={9}
                        fill="none"
                        stroke="none"
                        opacity={0}
                      />
                      <circle cx={PAD + point.x} cy={cy} r={5.4} fill="var(--surface)" />
                      <circle
                        cx={PAD + point.x}
                        cy={cy}
                        r={open ? 5 : 4}
                        fill={fill}
                        stroke={open ? "var(--ink)" : "var(--surface)"}
                        strokeWidth={open ? 2 : 1.4}
                      />
                    </>
                  );
                  if (!isActive) return <g key={point.id}>{dot}</g>;
                  const row0 = data.rows.find((r) => r.id === point.id)!;
                  return (
                    <g
                      key={point.id}
                      role="button"
                      tabIndex={0}
                      aria-pressed={open}
                      className="cursor-pointer"
                      onClick={() => onSelect(point.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          onSelect(point.id);
                        }
                      }}
                    >
                      <title>{`${row0.query}: ${signed(point.value)} nDCG@10. Open this query.`}</title>
                      {dot}
                    </g>
                  );
                })}
              </g>
            );
          })}

          <line
            x1={0}
            x2={WIDTH}
            y1={height - AXIS + 2}
            y2={height - AXIS + 2}
            stroke="var(--line-strong)"
            strokeWidth={1}
          />
          {TICKS.map((t) => (
            <text
              key={t}
              x={x(t)}
              y={height - AXIS + 20}
              textAnchor={t === -1 ? "start" : t === 1 ? "end" : "middle"}
              className="numeric"
              fontSize={12}
              fill="var(--ink-soft)"
            >
              {tick(t)}
            </text>
          ))}
          <text x={x(0)} y={height - 2} textAnchor="middle" fontSize={12} fill="var(--ink-soft)">
            nDCG@10 against BM25, one dot per query
          </text>
        </svg>
      </div>
    </figure>
  );
}
