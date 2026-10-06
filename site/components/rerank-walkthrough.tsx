"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftIcon, ArrowRightIcon } from "@phosphor-icons/react";
import rerankData from "@/data/rerank.json";
import { Term } from "@/components/term";
import {
  label,
  openingQuery,
  rerankerRows,
  runFacts,
  walkthroughFacts,
  type Detail,
  type Rerank,
} from "@/lib/rerank";
import { loadDetail } from "@/lib/rerank-detail";

const data = rerankData as Rerank;
const rows = rerankerRows(data);
const facts = runFacts(data);
const HELPED = rows[0].method;
const HURT = rows[rows.length - 1].method;
const QUERY = data.queries.find((q) => q.id === openingQuery(data, HELPED, HURT))!;
const STEPS = ["Search", "Read", "Sort", "Grade", "Compare"] as const;
const TOP = 10;

const at = (n: number | null) => (n === null ? "?" : String(n).padStart(2, "0"));

/**
 * Re-ranking, shown on one real question instead of described.
 *
 * Five verbs, each one thing the pipeline does: the search hands back its
 * shortlist, the re-ranker reads and scores every item, the list is sorted by
 * score, the order is graded, and a second model does the same with the
 * opposite result. The twenty rows are one list that only ever moves, so the
 * reader watches the answer travel rather than reading that it did. Every step
 * is a button, so the reader sets the pace and can go back.
 */
export function RerankWalkthrough() {
  const [step, setStep] = useState(0);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [failed, setFailed] = useState(false);
  const holder = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    let alive = true;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        loadDetail()
          .then((body) => alive && setDetail(body))
          .catch(() => alive && setFailed(true));
      },
      { rootMargin: "600px" },
    );
    observer.observe(node);
    return () => {
      alive = false;
      observer.disconnect();
    };
  }, []);

  const q = detail?.queries[QUERY.id];
  const story = useMemo(() => (q ? walkthroughFacts(QUERY, q, HELPED, HURT, data.floor) : null), [q]);

  // Which order the list is in, and whose scores it shows, at each step.
  const method = step >= 4 ? HURT : HELPED;
  const order = q ? (step <= 1 ? q.orders[data.floor] : q.orders[method]) : [];
  const searchOrder = q?.orders[data.floor] ?? [];
  const scoreOf = (doc: string) => {
    const ranked = q?.orders[method];
    const scores = q?.scores[method];
    if (!ranked || !scores) return null;
    return scores[ranked.indexOf(doc)] ?? null;
  };

  const captions: React.ReactNode[] = story
    ? [
        <>
          Someone asks <q className="italic">{QUERY.text}</q>. A fast search reads all{" "}
          {facts.documents?.toLocaleString()} abstracts and hands back its best {facts.topK}, like a librarian
          pulling a pile of books off the shelf in a hurry. The abstract a person marked as the answer is in
          the pile, at {at(story.from)}, but not on top.
        </>,
        <>
          Now the <Term id="rerank">re-ranker</Term> reads. <Term id="jev">{label(HELPED)}</Term> sees the
          question beside one abstract at a time and answers with a number from 0 to 4: how well does this
          abstract answer the question? {facts.topK} abstracts, {facts.topK} calls.
        </>,
        <>
          Sort the pile by those numbers. The answer climbs from {at(story.from)} to {at(story.helpedTo)}.
          Nothing was added and nothing removed: the same {facts.topK} abstracts, in a new order.
        </>,
        <>
          <Term id="ndcg">nDCG@10</Term> grades the top ten against the human judgements. With the answer at{" "}
          {at(story.from)} the order scored {story.floorNdcg.toFixed(3)}. At {at(story.helpedTo)} it scores{" "}
          {story.helpedNdcg.toFixed(3)}
          {story.helpedNdcg >= 0.9995 ? ", the best possible" : ""}.
        </>,
        <>
          Same {facts.topK} abstracts, scored by <Term id="laya">{label(HURT)}</Term> instead. The answer
          falls from {at(story.from)} to {at(story.hurtTo)}
          {story.hurtTo !== null && story.hurtTo > TOP ? ", out of the top ten," : ""} and the score drops to{" "}
          {story.hurtNdcg.toFixed(3)}. Over all {facts.queries} questions, that is the pattern in the
          standings above.
        </>,
      ]
    : [];

  return (
    <div ref={holder} className="grid min-w-0 gap-8 md:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] md:gap-12">
      <div className="grid content-start gap-6 md:sticky md:top-6 md:self-start">
        <ol className="flex flex-wrap gap-x-1 gap-y-2" aria-label="Steps">
          {STEPS.map((name, i) => (
            <li key={name}>
              <button
                type="button"
                onClick={() => setStep(i)}
                aria-current={i === step ? "step" : undefined}
                disabled={!story}
                className={`numeric min-h-11 px-2.5 text-micro underline-offset-[6px] ${
                  i === step
                    ? "text-ink underline decoration-ink decoration-2"
                    : "text-ink-soft hover:text-ink disabled:hover:text-ink-soft"
                }`}
              >
                {name}
              </button>
            </li>
          ))}
        </ol>

        {/* Above the caption, so the controls stay put while caption length changes. */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={!story || step === 0}
            className="numeric inline-flex min-h-11 items-center gap-2 border border-line bg-surface px-4 text-micro text-ink disabled:text-ink-soft disabled:opacity-50"
          >
            <ArrowLeftIcon size={16} weight="bold" aria-hidden />
            Back
          </button>
          <button
            type="button"
            onClick={() => setStep((s) => (s === STEPS.length - 1 ? 0 : s + 1))}
            disabled={!story}
            className="numeric inline-flex min-h-11 items-center gap-2 border border-ink bg-ink px-4 text-micro text-page disabled:opacity-50"
          >
            {step === STEPS.length - 1 ? "Start again" : STEPS[step + 1]}
            <ArrowRightIcon size={16} weight="bold" aria-hidden />
          </button>
        </div>

        <div className="min-h-[12rem]" aria-live="polite">
          {story ? (
            <p className="text-lead leading-relaxed">{captions[step]}</p>
          ) : (
            <p className="text-body text-ink-soft">
              {failed
                ? "The candidate file did not load, so there is nothing to walk through."
                : "Loading the question…"}
            </p>
          )}
          {story && step === 3 && (
            <p className="numeric mt-6 flex items-center gap-3 text-h2 leading-none">
              <span className="text-ink-soft">{story.floorNdcg.toFixed(3)}</span>
              <ArrowRightIcon size={24} weight="bold" aria-label="to" className="text-ink-soft" />
              <span className="text-up">{story.helpedNdcg.toFixed(3)}</span>
            </p>
          )}
          {story && step === 4 && (
            <p className="numeric mt-6 flex items-center gap-3 text-h2 leading-none">
              <span className="text-ink-soft">{story.floorNdcg.toFixed(3)}</span>
              <ArrowRightIcon size={24} weight="bold" aria-label="to" className="text-ink-soft" />
              <span className="text-down">{story.hurtNdcg.toFixed(3)}</span>
            </p>
          )}
        </div>

      </div>

      {/* The list. Rows are positioned by rank and only ever translate, so a
          change of order is a movement the eye can follow; under reduced motion
          the site-wide rule makes it a jump. */}
      <div className="relative min-w-0 [--row:2.25rem] sm:[--row:2.5rem]">
        <ol
          className="relative"
          style={{ height: `calc(var(--row) * ${facts.topK} + 2.5rem)` }}
          aria-label={`The ${facts.topK} abstracts for this question, in the current order`}
        >
          {story &&
            searchOrder.map((doc) => {
              const position = order.indexOf(doc);
              const answer = doc === story.answer;
              const score = scoreOf(doc);
              const showScore = step >= 1;
              return (
                <li
                  key={doc}
                  className="absolute inset-x-0 flex items-center gap-3 border-b border-line px-1 transition-transform duration-500 ease-[cubic-bezier(0.2,0.85,0.25,1)]"
                  style={{
                    height: "var(--row)",
                    transform: `translateY(calc(var(--row) * ${position} + ${position >= TOP ? "2.5rem" : "0rem"}))`,
                  }}
                >
                  <span className="numeric w-7 shrink-0 text-micro text-ink-soft">{at(position + 1)}</span>
                  <span
                    className={`min-w-0 flex-1 truncate text-micro ${answer ? "font-medium text-mark" : "text-ink"}`}
                  >
                    {data.titles[doc] ?? doc}
                  </span>
                  {answer && (
                    <span className="numeric shrink-0 border border-mark px-1 text-micro text-mark">
                      answer
                    </span>
                  )}
                  <span
                    className="numeric w-12 shrink-0 text-right text-micro text-ink-soft transition-opacity duration-300"
                    style={{
                      opacity: showScore ? 1 : 0,
                      transitionDelay: step === 1 ? `${searchOrder.indexOf(doc) * 40}ms` : "0ms",
                    }}
                    aria-hidden={!showScore}
                  >
                    {score === null ? "" : score.toFixed(2)}
                  </span>
                </li>
              );
            })}
          {!story &&
            Array.from({ length: facts.topK }, (_, i) => (
              <li
                key={i}
                className="absolute inset-x-0 flex items-center gap-3 border-b border-line px-1"
                style={{
                  height: "var(--row)",
                  transform: `translateY(calc(var(--row) * ${i} + ${i >= TOP ? "2.5rem" : "0rem"}))`,
                }}
                aria-hidden
              >
                <span className="h-2.5 w-7 bg-sunk" />
                <span className="h-2.5 flex-1 bg-sunk" style={{ maxWidth: `${55 + ((i * 37) % 40)}%` }} />
              </li>
            ))}
          <li
            aria-hidden
            className="numeric absolute inset-x-0 flex items-end border-b-2 border-ink pb-1 text-micro text-ink"
            style={{ top: `calc(var(--row) * ${TOP})`, height: "2.5rem" }}
          >
            Above this line: the top ten, all nDCG@10 grades
          </li>
        </ol>
      </div>
    </div>
  );
}
