"use client";

import { useState } from "react";
import rerankData from "@/data/rerank.json";
import { Term } from "@/components/term";
import type { Rerank } from "@/lib/rerank";
import { MEASURES, measureBoard, pairOn, type Measure } from "@/lib/rerank-story";

const data = rerankData as Rerank;
const WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const word = (n: number) => WORDS[n] ?? String(n);
const real = (ci: number[] | null) => !!ci && (ci[0] > 0 || ci[1] < 0);
const measures = MEASURES.filter((m) => measureBoard(data, m.key));

const SAY = {
  better: "Better than the search alone",
  worse: "Worse than the search alone",
  "no clear change": "Can't be told apart from the search alone",
} as const;
const TONE = { better: "text-up", worse: "text-down", "no clear change": "text-ink-soft" } as const;
const FILL = { better: "bg-up", worse: "bg-down", "no clear change": "bg-line-strong" } as const;

/**
 * What the run found, graded whichever way the reader likes. The headline
 * measure is first; every other one redraws the same four bars around the
 * search's own score, with each verdict read from that measure's own interval.
 * Jev against the cross-encoder follows the switch too, because the answer to
 * "is Jev better?" depends on which question you ask of the order.
 */
export function RerankStoryFound({
  yardstick,
  leader,
  children,
}: {
  yardstick: string;
  leader: string;
  /** What the finding costs, rendered on the server after the comparison. */
  children?: React.ReactNode;
}) {
  const [measure, setMeasure] = useState<Measure>(measures[0]);
  const board = measureBoard(data, measure.key)!;
  const values = [board.floor, ...board.rows.map((r) => r.value)];
  const pad = (Math.max(...values) - Math.min(...values)) * 0.15;
  const lo = Math.min(...values) - pad;
  const hi = Math.max(...values) + pad;
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const fmt = (v: number) => (measure.share ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
  const gap = (v: number) =>
    measure.share
      ? `${(Math.abs(v - board.floor) * 100).toFixed(1)} points`
      : `${Math.round(Math.abs(v / board.floor - 1) * 100)}%`;

  const leaderLabel = board.rows.find((r) => r.method === leader)?.label ?? leader;
  const yardLabel = board.rows.find((r) => r.method === yardstick)?.label ?? yardstick;
  const pair = pairOn(data, measure.key, leader, yardstick);
  const pairs = measures.map((m) => ({ m, p: pairOn(data, m.key, leader, yardstick) }));
  const ahead = pairs.filter(({ p }) => p && real(p.ci95) && p.mean > 0);
  const ties = pairs.filter(({ p }) => p && !real(p.ci95));
  const floorNdcg = measureBoard(data, "ndcg@10")!.floor;

  return (
    <section aria-labelledby="found" className="grid gap-10 border-t border-line pt-16 pb-12">
      <header className="grid max-w-[46rem] gap-3.5">
        <h2 id="found" className="text-balance text-h2 font-semibold leading-tight tracking-tight">
          What we found: one clear winner, and <em>two that made it worse</em>.
        </h2>
        <p className="text-lead text-ink-soft">
          The search&apos;s own order scores {floorNdcg.toFixed(3)} on the whole top ten. Here is where each re-ranker
          landed on the same {data.summary.queries} questions, and how that holds up graded {word(measures.length - 1)}{" "}
          other ways.
        </p>
      </header>

      <div>
        <h3 className="mb-3.5 text-lead font-semibold">Grade it any way you like</h3>
        <div className="mb-3.5 flex flex-wrap gap-1.5" role="group" aria-label="Grade by">
          {measures.map((m) => (
            <button
              key={m.key}
              type="button"
              aria-pressed={m.key === measure.key}
              onClick={() => setMeasure(m)}
              className={`min-h-11 border px-3 py-1.5 text-left text-sm leading-tight ${
                m.key === measure.key ? "border-ink bg-ink text-page" : "border-line-strong bg-surface"
              }`}
            >
              {m.label}
              <small className={`numeric mt-0.5 block text-[11px] ${m.key === measure.key ? "opacity-75" : "text-ink-soft"}`}>
                {m.tech}
              </small>
            </button>
          ))}
        </div>
        <p className="mb-3 max-w-[60ch] text-[1.05rem]" aria-live="polite">
          {measure.ask} The search&apos;s own order scores <span className="numeric font-medium">{fmt(board.floor)}</span>.
        </p>

        <div className="border border-line bg-surface px-4 sm:px-5.5">
          {board.rows.map((row) => {
            const up = row.value >= board.floor;
            return (
              <div
                key={row.method}
                className="grid grid-cols-[minmax(0,1fr)_4.2rem] items-center gap-x-3.5 gap-y-1.5 border-b border-line py-3.5 lg:grid-cols-[9rem_minmax(0,1fr)_4.2rem]"
              >
                <b className="text-lg">{row.label}</b>
                <div
                  role="img"
                  aria-label={`${row.label}: ${fmt(row.value)} against the search's ${fmt(board.floor)}`}
                  className="story-track relative col-span-2 h-[18px] lg:col-span-1"
                >
                  <span className="absolute -inset-y-1.5 border-l-2 border-ink" style={{ left: `${x(board.floor)}%` }} />
                  <i
                    className={`absolute top-[3px] h-3 ${FILL[row.verdict]}`}
                    style={{ left: `${x(Math.min(row.value, board.floor))}%`, width: `${Math.abs(x(row.value) - x(board.floor))}%` }}
                  />
                </div>
                <span className={`numeric row-start-1 col-start-2 text-right text-lg font-medium lg:col-start-3 ${TONE[row.verdict]}`}>
                  {fmt(row.value)}
                </span>
                <p className="col-span-2 text-[0.9375rem] leading-snug text-ink-soft lg:col-span-3">
                  <strong className="font-semibold text-ink">{SAY[row.verdict]}.</strong> {gap(row.value)}{" "}
                  {up ? "above" : "below"} it. Better on {row.paired.better} questions, worse on {row.paired.worse};{" "}
                  {row.paired.same} unchanged.
                </p>
              </div>
            );
          })}
          <div className="flex justify-between gap-3 py-2.5 text-micro text-ink-soft">
            <span>Left of the line: worse than the search alone</span>
            <span className="text-right">Right: better</span>
          </div>
        </div>
        <p className="mt-2.5 max-w-[70ch] text-micro text-ink-soft">
          Each re-ranker is compared with the search question by question, with a{" "}
          <Term id="interval">95% range</Term>. With {word(measures.length)} ways to grade, a borderline result can
          clear the bar by chance: the {yardLabel.toLowerCase()}&apos;s gain on the whole top ten is one of those.
        </p>
      </div>

      {pair && (
        <div className="grid max-w-[46rem] gap-2.5 text-lead">
          <h3 className="font-semibold">{leaderLabel} against the yardstick</h3>
          <p className="text-ink-soft">
            The {yardLabel.toLowerCase()} is the tool this job usually gets. Graded on{" "}
            <b className="text-ink">{measure.label.toLowerCase()}</b> and compared question by question, {leaderLabel} was
            better on {pair.better} questions and the {yardLabel.toLowerCase()} on {pair.worse}; {pair.same} were tied.{" "}
            <b className="text-ink">
              {real(pair.ci95)
                ? `On this measure, ${pair.mean > 0 ? leaderLabel : `the ${yardLabel.toLowerCase()}`} is really ahead.`
                : "On this measure, the two can't be told apart."}
            </b>
          </p>
          <p className="text-ink-soft">
            Across all {word(measures.length)} ways of grading, {leaderLabel} is ahead on {word(ahead.length)}.
            {ties.length > 0 &&
              ` On ${ties.map(({ m }) => m.label.toLowerCase()).join(" and ")}, they tie: ${leaderLabel}'s lead is in the rest of the top ten, not the very first place.`}
          </p>
        </div>
      )}
      {children}
    </section>
  );
}
