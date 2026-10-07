"use client";

import { useEffect, useRef, useState } from "react";
import type { StoryQuestion } from "@/lib/rerank-story";

const pad = (n: number) => String(n).padStart(2, "0");
const READ_EACH = 140;
const START = 600;

/**
 * The idea, shown once: a re-ranker reads the question beside each of the 20
 * abstracts, scores it, and the list is sorted by score. It is the best
 * re-ranker on the story's question, but the text says all four work this way,
 * and it ends by pointing at the one that buried the same answer.
 *
 * It plays when it scrolls into view and never on its own again; Play again
 * replays it. With reduced motion it goes straight to the sorted list.
 */
export function RerankStoryIdea({ story }: { story: StoryQuestion }) {
  const { helped, results, answer } = story;
  const search = results.map((r) => r.id);
  const title = Object.fromEntries(results.map((r) => [r.id, r.title]));
  const scoreOf = (id: string) => helped.scores[helped.order.indexOf(id)];
  const best = Math.max(...helped.scores) === scoreOf(answer);

  // How many of the 20 have been scored, which one is being read, and whether
  // the list has been sorted yet. Everything on screen follows from these three.
  const [read, setRead] = useState(0);
  const [reading, setReading] = useState<number | null>(null);
  const [sorted, setSorted] = useState(false);
  const [done, setDone] = useState(false);
  const board = useRef<HTMLDivElement>(null);
  const timers = useRef<number[]>([]);

  const play = () => {
    timers.current.forEach(clearTimeout);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const later = (fn: () => void, ms: number) => timers.current.push(window.setTimeout(fn, reduced ? 0 : ms));
    setRead(0);
    setReading(null);
    setSorted(false);
    setDone(false);
    search.forEach((_, i) => {
      later(() => setReading(i), START + i * READ_EACH);
      later(() => setRead(i + 1), START + i * READ_EACH + 80);
    });
    const all = START + search.length * READ_EACH + 200;
    later(() => setReading(null), all);
    later(() => setSorted(true), all + 1100);
    later(() => setDone(true), all + 2200);
  };

  useEffect(() => {
    const node = board.current;
    if (!node) return;
    const seen = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          seen.disconnect();
          play();
        }
      },
      { threshold: 0.35 },
    );
    seen.observe(node);
    const pending = timers.current;
    return () => {
      seen.disconnect();
      pending.forEach(clearTimeout);
    };
    // play is rebuilt each render but only ever reads the story, which does not change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const order = sorted ? helped.order : search;
  const status = done
    ? null
    : sorted
      ? "Sorting by score."
      : reading !== null
        ? `Reading ${reading + 1} of ${search.length}…`
        : read === search.length
          ? `It gave the right answer ${best ? "the highest score of the " + search.length : "a score of"}${best ? ": " : " "}${scoreOf(answer).toFixed(2)}. Now sort by score.`
          : `In the search's order, the right answer is at ${pad(story.searchAt)}.`;

  return (
    <section
      aria-labelledby="idea"
      className="grid gap-7 border-t border-line pt-16 pb-12 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] lg:grid-rows-[auto_1fr] lg:gap-x-14 lg:gap-y-5"
    >
      <div className="grid content-start gap-4 lg:col-start-1 lg:row-start-1">
        <h2 id="idea" className="text-balance text-h2 font-semibold leading-tight tracking-tight">
          So add a second reader: a <em>re-ranker</em>.
        </h2>
        <p className="max-w-[42ch] text-lead leading-relaxed text-ink-soft">
          The search is fast because it only compares words and meanings, across thousands of abstracts. A
          re-ranker is slower and more careful. It takes the {search.length} the search kept,{" "}
          <b className="font-semibold text-ink">reads the question beside each one</b>, scores how well that abstract
          answers it, and puts them back in order of score.
        </p>
        <p className="max-w-[42ch] text-lead leading-relaxed text-ink-soft">
          All four re-rankers we tested work exactly like this. They differ only in how well they score. Here is one
          of them, {helped.label}, on this question.
        </p>
      </div>

      <div ref={board} className="story-board min-w-0 lg:col-start-2 lg:row-span-2 lg:row-start-1" aria-hidden>
        <div className="absolute inset-x-0 top-0 grid h-[var(--head)] grid-cols-[minmax(0,1fr)_6.5rem] gap-2.5 pl-[2.4rem] text-micro text-ink-soft">
          <span>The {search.length} the search kept</span>
          <span className={`text-right transition-opacity ${read ? "opacity-100" : "opacity-0"}`}>Score</span>
        </div>
        {search.map((_, i) => (
          <span
            key={i}
            className="numeric absolute left-0 w-8 text-xs leading-[var(--row)] text-ink-soft"
            style={{ top: `calc(var(--head) + var(--row) * ${i})` }}
          >
            {pad(i + 1)}
          </span>
        ))}
        {search.map((id, i) => {
          const isAnswer = id === answer;
          const scored = search.indexOf(id) < read;
          return (
            <div
              key={id}
              className={`story-row absolute inset-x-0 top-[var(--head)] ml-[2.4rem] grid h-[calc(var(--row)-4px)] grid-cols-[minmax(0,1fr)_6.5rem] items-center gap-2.5 border pl-2.5 text-[0.9rem] ${
                isAnswer ? "z-10 border-mark bg-mark-wash" : reading === i ? "border-ink bg-surface" : "border-line bg-surface"
              }`}
              style={{
                transform: `translateY(calc(var(--row) * ${order.indexOf(id)}))`,
                transitionDelay: sorted ? `${order.indexOf(id) * 25}ms` : "0ms",
              }}
            >
              <span className={`truncate ${isAnswer ? "font-semibold" : ""}`}>{title[id]}</span>
              <span className="grid grid-cols-[minmax(0,1fr)_2.4rem] items-center gap-1.5 pr-2">
                <span className="relative h-1.5 bg-sunk">
                  <i
                    className={`story-bar absolute inset-y-0 left-0 ${isAnswer ? "bg-mark" : "bg-line-strong"}`}
                    style={{ width: scored ? `${(scoreOf(id) / Math.max(4, ...helped.scores)) * 100}%` : 0 }}
                  />
                </span>
                <span
                  className={`numeric text-right text-xs transition-opacity ${scored ? "opacity-100" : "opacity-0"} ${
                    isAnswer ? "font-medium text-ink" : "text-ink-soft"
                  }`}
                >
                  {scoreOf(id).toFixed(2)}
                </span>
              </span>
            </div>
          );
        })}
      </div>

      <div className="grid content-start gap-3 lg:col-start-1 lg:row-start-2">
        <p className="min-h-[3.2em] text-[1.05rem]" aria-live="polite">
          {done ? (
            <>
              The right answer moves from {pad(story.searchAt)} to {pad(helped.at)}. Nothing was added or removed: the
              same {search.length}, in a better order.
              <span className="mt-2.5 block text-ink-soft">
                But they don&apos;t all get it right. On this same question, another of the four buried the right answer
                at {pad(story.buried.at)}.
              </span>
            </>
          ) : (
            status
          )}
        </p>
        <button
          type="button"
          onClick={play}
          disabled={!done}
          className="min-h-11 justify-self-start border border-line-strong bg-surface px-4 text-[0.9375rem] disabled:cursor-default disabled:opacity-40"
        >
          Play again
        </button>
      </div>
    </section>
  );
}
