"use client";

import { useEffect, useRef, useState } from "react";
import type { Passage } from "@/lib/example";
import type { SlopeLine } from "@/lib/slope";

const ROW = 30;
const DURATION = 420;

const easeOut = (t: number) => 1 - (1 - t) ** 3;

/** Where a passage sits, in pixels from the top of the chart. */
const y = (position: number) => (position - 0.5) * ROW;

function strokeOf(moved: number) {
  if (moved > 0) return "var(--up)";
  if (moved < 0) return "var(--down)";
  return "var(--line-strong)";
}

/**
 * BM25's twenty positions on the left, the re-ranker's ordering of the same
 * twenty on the right, one line per passage.
 *
 * Changing the method animates the lines into their new positions rather than
 * cutting to them, because the thing being described is a movement: a cut leaves
 * the reader to diff two pictures from memory. The tween drives the connector
 * lines and the right-hand rows from one set of numbers, so they cannot drift
 * apart mid-flight, and `prefers-reduced-motion` gets the end state with no
 * flight at all.
 */
export function SlopeChart({
  lines,
  passages,
  scores,
  failed,
  methodLabel,
  open,
  onOpen,
}: {
  lines: SlopeLine[];
  passages: Record<string, Passage>;
  scores: Record<string, number | null>;
  failed: string[];
  methodLabel: string;
  open: string | null;
  onOpen: (docId: string) => void;
}) {
  const target = lines.map((line) => line.to);
  const key = target.join(",");
  const [shown, setShown] = useState(target);
  const shownRef = useRef(target);

  useEffect(() => {
    const next = key.split(",").map(Number);
    const from = shownRef.current;
    const still =
      typeof window === "undefined" ||
      !window.matchMedia("(prefers-reduced-motion: no-preference)").matches ||
      from.length !== next.length;
    if (still) {
      shownRef.current = next;
      setShown(next);
      return;
    }
    let frame = 0;
    const started = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - started) / DURATION);
      const at = next.map((to, i) => from[i] + (to - from[i]) * easeOut(t));
      shownRef.current = at;
      setShown(at);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [key]);

  const height = lines.length * ROW;

  return (
    <div className="grid min-w-0 gap-2">
      <div className="grid grid-cols-[98px_minmax(0,1fr)_minmax(0,10.75rem)] gap-0 text-[11px] sm:grid-cols-[minmax(0,1fr)_120px_minmax(0,1fr)] sm:text-micro">
        <h4 className="numeric pb-2 font-medium text-ink-soft">BM25</h4>
        <p className="pb-2" aria-hidden />
        <h4 className="numeric pb-2 text-right font-medium text-ink">{methodLabel}</h4>

        <ol className="relative m-0 list-none p-0" style={{ height }}>
          {lines.map((line) => (
            <li
              key={line.docId}
              className="absolute inset-x-0 flex items-center gap-1.5 pr-2 sm:gap-2"
              style={{ top: (line.from - 1) * ROW, height: ROW }}
            >
              <span className="numeric w-4 shrink-0 text-right text-ink-soft sm:w-5">{line.from}</span>
              <Marker relevant={line.relevant} grade={line.grade} />
              <span className="min-w-0 truncate text-ink-soft">
                <span className="numeric">{line.docId}</span>
                <span className="hidden sm:inline"> {passages[line.docId]?.title}</span>
              </span>
            </li>
          ))}
        </ol>

        <svg
          viewBox={`0 0 100 ${height}`}
          preserveAspectRatio="none"
          aria-hidden
          style={{ height }}
          className="w-full"
        >
          {lines.map((line, i) => (
            <line
              key={line.docId}
              x1={0}
              y1={y(line.from)}
              x2={100}
              y2={y(shown[i] ?? line.to)}
              stroke={strokeOf(line.moved)}
              strokeWidth={line.relevant ? 2 : 1}
              opacity={line.relevant ? 1 : 0.5}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>

        <ol className="relative m-0 list-none p-0" style={{ height }}>
          {lines.map((line, i) => {
            const score = scores[line.docId];
            const isOpen = open === line.docId;
            const didFail = failed.includes(line.docId);
            return (
              <li
                key={line.docId}
                className="absolute inset-x-0"
                style={{ top: 0, height: ROW, transform: `translateY(${((shown[i] ?? line.to) - 1) * ROW}px)` }}
              >
                <button
                  type="button"
                  onClick={() => onOpen(line.docId)}
                  aria-expanded={isOpen}
                  className={`flex h-full w-full items-center gap-1.5 pl-2 text-left sm:gap-2 ${
                    isOpen ? "bg-mark-wash" : "hover:bg-sunk"
                  }`}
                >
                  <span className="numeric w-4 shrink-0 text-right text-ink-soft sm:w-5">{line.to}</span>
                  <Marker relevant={line.relevant} grade={line.grade} />
                  <span className="min-w-0 flex-1 truncate">
                    <span className="numeric">{line.docId}</span>
                    <span className="hidden sm:inline text-ink-soft"> {passages[line.docId]?.title}</span>
                  </span>
                  <span className="numeric shrink-0 text-right text-ink-soft">
                    {didFail ? "held" : typeof score === "number" ? score.toFixed(2) : "—"}
                  </span>
                  <span
                    className="numeric w-9 shrink-0 text-right"
                    style={{ color: strokeOf(line.moved) }}
                  >
                    {line.moved === 0 ? "—" : `${line.moved > 0 ? "↑" : "↓"}${Math.abs(line.moved)}`}
                    <span className="sr-only">
                      {line.moved === 0
                        ? " unchanged"
                        : ` ${Math.abs(line.moved)} places ${line.moved > 0 ? "up" : "down"}`}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

function Marker({ relevant, grade }: { relevant: boolean; grade: number }) {
  if (!relevant) {
    return <span aria-hidden className="h-1.5 w-1.5 shrink-0 border border-line-strong" />;
  }
  return (
    <>
      <span aria-hidden className="h-1.5 w-1.5 shrink-0 bg-mark" />
      <span className="sr-only">judged relevant, grade {grade}, </span>
    </>
  );
}
