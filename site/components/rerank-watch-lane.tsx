"use client";

import { useEffect, useRef, useState } from "react";

const TOP = 10;
// Normal stays inside the motion budget (500 ms of travel, at most 250 ms of
// stagger across all 20); slow is for following one candidate by eye.
export const SPEED = { normal: { travel: 500, stagger: 250 }, slow: { travel: 1800, stagger: 1300 } } as const;
export type Speed = keyof typeof SPEED;

const pad = (n: number) => String(n).padStart(2, "0");
const slotY = (pos0: number) => `calc(var(--row) * ${pos0} + ${pos0 >= TOP ? "var(--divider)" : "0px"})`;
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const nextFrame = (fn: () => void) => requestAnimationFrame(() => requestAnimationFrame(fn));

/**
 * The 20 candidates as one list that only ever moves. Slots and their numbers
 * stay put; each candidate travels from where it stands to where the ranking
 * being watched put it, a moment after the one above it, so the eye can follow
 * one. Choosing a ranking moves the list from its current order (Jev's turning
 * into Laya's); a replay starts again from the search's. The list waits until
 * it is on screen before it moves.
 */
export function RerankWatchLane({
  queryId,
  orders,
  floor,
  watching,
  answerIds,
  lit,
  onToggle,
  titles,
  name,
  speed,
  replay,
  skip,
  onAnimating,
}: {
  queryId: string;
  orders: Record<string, string[]>;
  floor: string;
  watching: string;
  answerIds: Map<string, string>;
  lit: Set<string>;
  onToggle: (doc: string) => void;
  titles: Record<string, string>;
  name: (method: string) => string;
  speed: Speed;
  /** Bumped by the parent to replay from the search's order. */
  replay: number;
  /** Bumped by the parent to jump to the end of the move in progress. */
  skip: number;
  onAnimating: (moving: boolean) => void;
}) {
  const [view, setView] = useState<{ shown: string; instant: boolean }>({ shown: floor, instant: true });
  const [spoken, setSpoken] = useState("");
  const lane = useRef<HTMLDivElement>(null);
  const visible = useRef(false);
  const pending = useRef<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const frames = useRef<number[]>([]);

  const { travel, stagger } = SPEED[speed];

  // Every state change happens in an animation frame: the reset has to be
  // painted before the move starts, or there is nothing to travel from.
  const later = (fn: () => void) => frames.current.push(requestAnimationFrame(fn));

  const play = () => {
    nextFrame(() => {
      const target = pending.current;
      if (target === null || !visible.current) return;
      pending.current = null;
      setView({ shown: target, instant: false });
      if (reduced()) {
        lane.current?.classList.remove("fade");
        void lane.current?.offsetWidth;
        lane.current?.classList.add("fade");
      }
      onAnimating(!reduced());
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => finish(target), reduced() ? 220 : travel + stagger + 40);
    });
  };

  const finish = (m: string) => {
    onAnimating(false);
    const placed = [...answerIds.entries()].slice(0, 3).map(([doc, id]) => `${id} at ${orders[m].indexOf(doc) + 1}`);
    setSpoken(m === floor ? "The search's own order." : `${name(m)}'s order. ${placed.join(", ")}${answerIds.size > 3 ? ", and more" : ""}.`);
  };

  const moveTo = (target: string, fromSearch: boolean) => {
    if (fromSearch) later(() => setView({ shown: floor, instant: true }));
    pending.current = target;
    play();
  };

  // A new question starts from the search's order.
  useEffect(() => {
    moveTo(watching, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryId]);

  // A new ranking continues from wherever the list stands.
  useEffect(() => {
    moveTo(watching, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [watching]);

  useEffect(() => {
    if (replay) moveTo(watching, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replay]);

  useEffect(() => {
    if (!skip) return;
    later(() => {
      window.clearTimeout(timer.current);
      setView((v) => ({ ...v, instant: true }));
      finish(view.shown);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [skip]);

  useEffect(() => {
    const node = lane.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        visible.current = entries.some((e) => e.isIntersecting);
        if (visible.current && pending.current !== null) play();
      },
      { threshold: 0.2 },
    );
    observer.observe(node);
    const owned = frames.current;
    return () => {
      observer.disconnect();
      owned.forEach(cancelAnimationFrame);
      window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const searchOrder = orders[floor];
  const order = orders[view.shown] ?? searchOrder;

  return (
    <>
      <div
        ref={lane}
        className={`watch-lane ${view.instant ? "still" : ""}`}
        style={{ "--travel": `${travel}ms` } as React.CSSProperties}
      >
        {Array.from({ length: 20 }, (_, i) => (
          <div
            key={i}
            aria-hidden
            className="numeric absolute left-0 grid w-7 items-center text-micro text-ink-soft"
            style={{ top: slotY(i), height: "var(--row)" }}
          >
            {pad(i + 1)}
          </div>
        ))}
        <div
          aria-hidden
          className="absolute inset-x-0 flex items-center gap-2.5"
          style={{ top: `calc(var(--row) * ${TOP})`, height: "var(--divider)" }}
        >
          <span className="flex-1 border-t-2 border-ink" />
          <span className="text-micro text-ink">Top ten: only these are scored</span>
          <span className="flex-1 border-t-2 border-ink" />
        </div>

        {searchOrder.map((doc) => {
          const to = order.indexOf(doc);
          const from = searchOrder.indexOf(doc);
          const id = answerIds.get(doc);
          const on = lit.has(doc);
          const moved = from - to;
          const labelled = view.shown !== floor && (id || on);
          return (
            <button
              key={doc}
              type="button"
              onClick={() => onToggle(doc)}
              aria-pressed={on}
              aria-label={`${titles[doc] ?? doc}${id ? `, answer ${id}` : ""}, position ${to + 1}. ${on ? "Highlighted." : "Tap to highlight."}`}
              title={titles[doc] ?? doc}
              className={`watch-cand absolute left-[34px] right-0 top-0 mt-0.5 flex min-w-0 items-center gap-2.5 border bg-surface px-2.5 text-left text-[14px] leading-[1.3] ${
                id ? "z-[2] border-mark font-semibold text-mark" : "border-line text-ink-soft"
              } ${on ? "lit z-[3]" : ""}`}
              style={{
                height: "calc(var(--row) - 4px)",
                transform: `translateY(${slotY(to)})`,
                transitionDelay: view.instant ? "0ms" : `${Math.round((to * stagger) / 19)}ms`,
              }}
            >
              {id && <span className="numeric shrink-0 bg-mark px-1.5 text-[11px] font-semibold leading-[1.6] text-page">{id}</span>}
              <span className="line-clamp-2 min-w-0 flex-1">{titles[doc] ?? doc}</span>
              {labelled && (
                <span className="numeric shrink-0 whitespace-nowrap text-[12px]" aria-hidden>
                  {moved === 0 ? (
                    "stayed"
                  ) : (
                    <>
                      <span className="hidden min-[401px]:inline">
                        {moved > 0 ? "up" : "down"} from {pad(from + 1)}
                      </span>
                      <span className="min-[401px]:hidden">was {pad(from + 1)}</span>
                    </>
                  )}
                </span>
              )}
            </button>
          );
        })}
      </div>
      <p className="sr-only" aria-live="polite">
        {spoken}
      </p>
    </>
  );
}
