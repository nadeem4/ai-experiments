"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import rerankData from "@/data/rerank.json";
import { RerankWatchLane, type Speed } from "@/components/rerank-watch-lane";
import { RerankWatchPicker } from "@/components/rerank-watch-picker";
import { queryNdcg, signed, type Detail, type Rerank } from "@/lib/rerank";
import { loadDetail } from "@/lib/rerank-detail";
import { answerOrder, cellIdsFit, questionMeasures, verdictLine } from "@/lib/rerank-watch";

const data = rerankData as Rerank;
const FLOOR = data.floor;
const TOP = 10;

// The section's own short names: a strip has room for a name and a few words.
const NAMES: Record<string, { name: string; kind: string; short: string }> = {
  [FLOOR]: { name: "The search", kind: "no re-ranking", short: "Search" },
  "jev-score": { name: "Jev", kind: "hosted model", short: "Jev" },
  "cross-encoder": { name: "Cross-encoder", kind: "MiniLM", short: "Cross" },
  "laya-score": { name: "Laya", kind: "open weights", short: "Laya" },
  "laya-typed-score": { name: "Laya typed", kind: "fine-tuned Laya", short: "Typed" },
};
const name = (m: string) => NAMES[m]?.name ?? m;
const pad = (n: number) => String(n).padStart(2, "0");
const toneOf = (v: number) => (v > 1e-9 ? "text-up" : v < -1e-9 ? "text-down" : "text-ink-soft");

/**
 * Watch one question being re-ranked.
 *
 * The question and its answers sit in a panel that stays beside the board on
 * wide screens; on phones a slim bar keeps the question and the controls on
 * screen while the list moves. A sentence says what happened, five strips
 * compare every ranking at once, one list moves into whichever ranking is
 * tapped, and this question's scores close it. Tapping any candidate, answer
 * or card row highlights it everywhere; tapping it again clears it.
 */
export function RerankWatch({
  queryId,
  onQuery,
  watching,
  onWatch,
}: {
  queryId: string;
  onQuery: (id: string) => void;
  watching: string;
  onWatch: (method: string) => void;
}) {
  const query = data.queries.find((q) => q.id === queryId) ?? data.queries[0];
  const [detail, setDetail] = useState<Detail | null>(null);
  const [failed, setFailed] = useState(false);
  const [lit, setLit] = useState<Set<string>>(new Set());
  const [litFor, setLitFor] = useState(queryId);
  const [speed, setSpeed] = useState<Speed>("normal");
  const [replay, setReplay] = useState(0);
  const [skip, setSkip] = useState(0);
  const [moving, setMoving] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  const sheetId = useId();

  // Highlights belong to one question. Reset during render, not in an effect,
  // so the new question never draws with the old one's rings.
  if (litFor !== queryId) {
    setLitFor(queryId);
    setLit(new Set());
  }

  useEffect(() => {
    const node = root.current;
    if (!node) return;
    let alive = true;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
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

  // The list leaves room for the phone's sticky bar at whatever height it wraps to.
  useEffect(() => {
    const node = bar.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) =>
      root.current?.style.setProperty("--watch-bar", `${Math.ceil(entry.target.getBoundingClientRect().height)}px`),
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const orders = detail?.queries[query.id]?.orders ?? null;
  const rankings = useMemo(() => (orders ? Object.keys(NAMES).filter((m) => orders[m]) : []), [orders]);
  const rerankers = rankings.filter((m) => m !== FLOOR);
  const answers = useMemo(() => (orders ? answerOrder(query.relevant, orders[FLOOR]) : query.relevant), [orders, query]);
  const answerIds = useMemo(() => new Map(answers.map((d, i) => [d, `A${i + 1}`])), [answers]);
  const ndcg = useMemo(
    () => Object.fromEntries(Object.keys(NAMES).map((m) => [m, queryNdcg(query, m, FLOOR).value ?? 0])),
    [query],
  );
  const toggle = (doc: string) =>
    setLit((now) => {
      const next = new Set(now);
      if (next.has(doc)) next.delete(doc);
      else next.add(doc);
      return next;
    });
  const title = (doc: string) => data.titles[doc] ?? doc;

  const controls = (
    <div className="flex flex-wrap items-center gap-2">
      <div className="inline-flex border border-line-strong" role="group" aria-label="Speed">
        {(["normal", "slow"] as const).map((s) => (
          <button
            key={s}
            type="button"
            aria-pressed={speed === s}
            onClick={() => {
              setSpeed(s);
              setReplay((n) => n + 1);
            }}
            className={`min-h-11 px-3.5 text-micro ${speed === s ? "bg-ink text-page" : "bg-surface"}`}
          >
            {s === "normal" ? "Normal" : "Slow"}
          </button>
        ))}
      </div>
      {moving ? (
        <button type="button" onClick={() => setSkip((n) => n + 1)} className="min-h-11 border border-line-strong bg-surface px-4 text-micro">
          Skip to end
        </button>
      ) : (
        <button type="button" onClick={() => setReplay((n) => n + 1)} className="min-h-11 border border-ink bg-ink px-4 text-micro text-page">
          Replay
        </button>
      )}
    </div>
  );

  return (
    <div ref={root} className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-8 lg:grid-cols-[21rem_minmax(0,1fr)] lg:items-start lg:gap-10">
      {/* ---- the question, its answers, and where each sits ---- */}
      <div className="grid min-w-0 content-start gap-3 lg:sticky lg:top-4 lg:max-h-[calc(100dvh-2rem)] lg:overflow-y-auto">
        <RerankWatchPicker queries={data.queries} selected={query} onSelect={onQuery} sheetId={sheetId} />
        <Answers answers={answers} answerIds={answerIds} lit={lit} onToggle={toggle} title={title} />
        <PositionsCard
          answers={answers}
          answerIds={answerIds}
          lit={lit}
          onToggle={toggle}
          orders={orders}
          rankings={rankings}
          watching={watching}
          title={title}
        />
      </div>

      {/* ---- the stage ---- */}
      <div className="grid min-w-0 content-start gap-9">
        <div
          ref={bar}
          className="sticky top-0 z-[5] grid min-w-0 gap-1.5 border-b border-line-strong bg-page pb-2.5 pt-2 lg:hidden"
          aria-label="Now watching"
        >
          <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
            <span className="truncate font-semibold">{query.text}</span>
            <button type="button" popoverTarget={sheetId} className="min-h-11 border border-line bg-surface px-3 text-micro">
              Change
            </button>
          </div>
          {controls}
        </div>

        {failed ? (
          <p className="max-w-[62ch] text-body text-ink-soft">The rankings did not load, so there is nothing to show for this question.</p>
        ) : !orders ? (
          <p className="text-body text-ink-soft">Loading the rankings…</p>
        ) : (
          <>
            <p className="max-w-[60ch] text-lead leading-relaxed">
              {verdictLine(orders, query.relevant, ndcg, FLOOR, rerankers, name).map((part, i) =>
                part.strong ? <b key={i} className="font-semibold">{part.text}</b> : <span key={i}>{part.text}</span>,
              )}
            </p>

            <section className="grid min-w-0 gap-3" aria-labelledby="watch-glance">
              <h3 id="watch-glance" className="text-lead font-semibold">
                Where each ranking put the answers
              </h3>
              <p className="max-w-[62ch] text-body text-ink-soft">
                Position 1 is on the left. Violet cells are the answers, numbered A1, A2 and so on by where the search
                put them. The black line closes the top ten. Tap a ranking to watch it.
              </p>
              <Strips
                orders={orders}
                rankings={rankings}
                answerIds={answerIds}
                lit={lit}
                ndcg={ndcg}
                watching={watching}
                onWatch={onWatch}
              />
            </section>

            <section className="grid min-w-0 gap-3" aria-labelledby="watch-list">
              <h3 id="watch-list" className="text-lead font-semibold">
                {watching === FLOOR ? "The search's own order" : `Watch ${name(watching)} reorder the 20`}
              </h3>
              <div className="hidden flex-wrap items-center justify-between gap-2 lg:flex">
                <p className="text-body text-ink-soft">
                  {watching === FLOOR ? "Before any re-ranking." : `${name(watching)}'s order, from the search's.`}
                </p>
                {controls}
              </div>
              <RerankWatchLane
                queryId={query.id}
                orders={orders}
                floor={FLOOR}
                watching={watching}
                answerIds={answerIds}
                lit={lit}
                onToggle={toggle}
                titles={data.titles}
                name={name}
                speed={speed}
                replay={replay}
                skip={skip}
                onAnimating={setMoving}
              />
              <p className="text-micro text-ink-soft">Tap any candidate or answer to highlight it in every ranking. Tap it again to clear it.</p>
            </section>

            <Scores watching={watching} relevant={query.relevant} orders={orders} ndcg={ndcg} />
          </>
        )}
      </div>
    </div>
  );
}

function Answers({
  answers,
  answerIds,
  lit,
  onToggle,
  title,
}: {
  answers: string[];
  answerIds: Map<string, string>;
  lit: Set<string>;
  onToggle: (doc: string) => void;
  title: (doc: string) => string;
}) {
  if (!answers.length) {
    return <p className="bg-sunk px-3.5 py-3 text-body">None of the 20 candidates the search fetched answers this question.</p>;
  }
  return (
    <div className="grid gap-2 border border-line bg-surface px-3.5 py-3">
      <p className="text-micro text-ink-soft">
        {answers.length === 1 ? "1 answer. Tap it to highlight it." : `${answers.length} answers. Tap any to highlight it.`}
      </p>
      {/* As tall as it needs up to about four and a half answers, then it
          scrolls: cutting the fifth in half is the cue that there is more. */}
      <div className="-mx-2 flex max-h-[40dvh] flex-col gap-0.5 overflow-y-auto overscroll-contain px-2 lg:max-h-[17rem]">
        {answers.map((doc, i) => {
          const twin = answers.slice(0, i).find((o) => title(o) === title(doc));
          return (
            <button
              key={doc}
              type="button"
              onClick={() => onToggle(doc)}
              aria-pressed={lit.has(doc)}
              className={`-mx-2 grid min-h-11 shrink-0 grid-cols-[auto_minmax(0,1fr)] items-baseline gap-2.5 border-[1.5px] p-2 text-left text-body leading-snug ${
                lit.has(doc) ? "border-ink bg-page" : "border-transparent hover:border-line-strong"
              }`}
            >
              <span className="numeric self-center bg-mark px-1.5 text-[11px] font-semibold leading-[1.6] text-page">{answerIds.get(doc)}</span>
              <span className="font-semibold text-mark">
                {title(doc)}
                {twin && (
                  <small className="mt-0.5 block text-micro font-normal text-ink-soft">
                    Same title as {answerIds.get(twin)}; a separate abstract in the dataset.
                  </small>
                )}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PositionsCard({
  answers,
  answerIds,
  lit,
  onToggle,
  orders,
  rankings,
  watching,
  title,
}: {
  answers: string[];
  answerIds: Map<string, string>;
  lit: Set<string>;
  onToggle: (doc: string) => void;
  orders: Record<string, string[]> | null;
  rankings: string[];
  watching: string;
  title: (doc: string) => string;
}) {
  if (!orders) return null;
  // A highlighted candidate that is not an answer gets a row while it is highlighted.
  const extra = [...lit].filter((d) => !answerIds.has(d)).sort((a, b) => orders[FLOOR].indexOf(a) - orders[FLOOR].indexOf(b));
  const rows = [...answers, ...extra];
  const label = (d: string) => answerIds.get(d) ?? `#${pad(orders[FLOOR].indexOf(d) + 1)}`;
  return (
    <div className="grid gap-2.5 border border-line bg-surface p-3.5" aria-live="polite">
      <h4 className="text-body font-semibold">Where each answer sits</h4>
      {rows.length === 0 ? (
        <p className="text-micro text-ink-soft">No answer to place. Tap any candidate in the list to see where it sits in each ranking.</p>
      ) : (
        <div className="max-h-[17rem] overflow-auto overscroll-contain">
          <table className="numeric w-full border-collapse text-[14px]">
            <thead>
              <tr>
                <th scope="col" className="sticky top-0 bg-surface">
                  <span className="sr-only">Candidate</span>
                </th>
                {rankings.map((m) => (
                  <th
                    key={m}
                    scope="col"
                    title={name(m)}
                    className={`sticky top-0 border-b border-line-strong px-1.5 py-1 text-right font-serif text-[12px] font-semibold ${
                      m === watching ? "bg-sunk text-ink" : "bg-surface text-ink-soft"
                    }`}
                  >
                    {NAMES[m].short}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((d) => {
                const on = lit.has(d);
                const ring = on ? "shadow-[inset_0_2px_0_var(--ink),inset_0_-2px_0_var(--ink)]" : "";
                return (
                  <tr key={d} className="border-t border-line first:border-t-0">
                    <th scope="row" className={`text-left ${ring}`}>
                      <button
                        type="button"
                        onClick={() => onToggle(d)}
                        aria-pressed={on}
                        title={title(d)}
                        className="min-h-8 min-w-11 px-1 py-0.5 text-left"
                      >
                        <span className={`px-1.5 text-[11px] font-semibold leading-[1.6] text-page ${answerIds.has(d) ? "bg-mark" : "bg-ink-soft"}`}>
                          {label(d)}
                        </span>
                      </button>
                    </th>
                    {rankings.map((m) => (
                      <td key={m} className={`px-1.5 py-1 text-right ${m === watching ? "bg-sunk font-semibold text-ink" : ""} ${ring}`}>
                        {pad(orders[m].indexOf(d) + 1)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {extra.length > 0 && (
        <p className="text-micro text-ink-soft">
          {extra.map((d) => (
            <span key={d} className="block">
              {label(d)}: {title(d)}, not an answer
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

function Strips({
  orders,
  rankings,
  answerIds,
  lit,
  ndcg,
  watching,
  onWatch,
}: {
  orders: Record<string, string[]>;
  rankings: string[];
  answerIds: Map<string, string>;
  lit: Set<string>;
  ndcg: Record<string, number>;
  watching: string;
  onWatch: (method: string) => void;
}) {
  const numbered = cellIdsFit(answerIds.size);
  return (
    <div className="grid gap-1.5" role="group" aria-label="Rankings">
      {rankings.map((m) => {
        const delta = ndcg[m] - ndcg[FLOOR];
        const where = [...answerIds.entries()].map(([d, id]) => `${id} at ${orders[m].indexOf(d) + 1}`).join(", ");
        return (
          <button
            key={m}
            type="button"
            aria-pressed={m === watching}
            onClick={() => onWatch(m)}
            aria-label={`${name(m)}, nDCG@10 ${ndcg[m].toFixed(3)}${where ? `. ${where}` : ""}. Tap to watch.`}
            className={`grid min-h-[52px] w-full grid-cols-[4.8rem_minmax(0,1fr)_3.5rem] items-center gap-2 border bg-surface px-2.5 py-2 text-left min-[481px]:grid-cols-[6.5rem_minmax(0,1fr)_4.2rem] min-[481px]:gap-2.5 ${
              m === watching ? "border-ink shadow-[inset_0_0_0_1px_var(--ink)]" : "border-line hover:border-line-strong"
            }`}
          >
            <span className="text-body font-semibold leading-tight">
              {name(m)}
              <small className="mt-0.5 hidden whitespace-nowrap text-micro font-normal text-ink-soft min-[481px]:block">{NAMES[m].kind}</small>
            </span>
            <span className="watch-cells" aria-hidden>
              {orders[m].map((doc, i) => (
                <Cell key={doc} id={answerIds.get(doc)} top={i < TOP} lit={lit.has(doc)} numbered={numbered} divider={i === TOP - 1} />
              ))}
            </span>
            <span className={`numeric text-right text-body leading-tight ${m === FLOOR ? "" : toneOf(delta)}`}>
              {ndcg[m].toFixed(3)}
              <small className="block text-micro">{m === FLOOR ? "nDCG@10" : signed(delta, 3)}</small>
            </span>
          </button>
        );
      })}
      <div className="grid grid-cols-[4.8rem_minmax(0,1fr)_3.5rem] gap-2 px-2.5 min-[481px]:grid-cols-[6.5rem_minmax(0,1fr)_4.2rem] min-[481px]:gap-2.5" aria-hidden>
        <span />
        <span className="numeric flex justify-between text-micro text-ink-soft">
          <span>1</span>
          <span>10 | 11</span>
          <span>20</span>
        </span>
        <span />
      </div>
    </div>
  );
}

function Cell({ id, top, lit, numbered, divider }: { id?: string; top: boolean; lit: boolean; numbered: boolean; divider: boolean }) {
  return (
    <>
      <i className={[id ? "ans" : top ? "top" : "", lit ? "lit" : ""].join(" ").trim()}>{id && numbered ? id.slice(1) : ""}</i>
      {divider && <b />}
    </>
  );
}

function Scores({
  watching,
  relevant,
  orders,
  ndcg,
}: {
  watching: string;
  relevant: string[];
  orders: Record<string, string[]>;
  ndcg: Record<string, number>;
}) {
  const cols = watching === FLOOR ? [FLOOR] : [FLOOR, watching];
  const ms = cols.map((m) => ({ ...questionMeasures(orders[m], relevant), ndcg: ndcg[m], delta: ndcg[m] - ndcg[FLOOR] }));
  const none = relevant.length ? "none" : "nothing to find";
  const rows: { name: string; tech: string; head?: boolean; cells: (string | React.ReactNode)[] }[] = [
    {
      name: "Ranking quality",
      tech: "nDCG@10",
      head: true,
      cells: ms.map((m, i) => <span key={i} className={i ? toneOf(m.delta) : ""}>{m.ndcg.toFixed(3)}</span>),
    },
    ...(cols.length > 1
      ? [{ name: "Change against the search", tech: "ΔnDCG@10", cells: ["baseline", <span key="d" className={toneOf(ms[1].delta)}>{signed(ms[1].delta, 3)}</span>] }]
      : []),
    { name: "Where the first answer landed", tech: "rank of the first relevant candidate", cells: ms.map((m) => (m.first ? pad(m.first) : none)) },
    { name: "First answer, as 1 ÷ its position", tech: "reciprocal rank, RR@10", cells: ms.map((m) => (relevant.length ? m.rr.toFixed(3) : none)) },
    { name: "Answers in the top ten", tech: "relevant candidates in the top 10", cells: ms.map((m) => (relevant.length ? `${m.inTop} of ${relevant.length}` : none)) },
  ];
  return (
    <section className="grid min-w-0 gap-3" aria-labelledby="watch-scored">
      <h3 id="watch-scored" className="text-lead font-semibold">
        How this question scored
      </h3>
      <p className="text-body text-ink-soft">
        {watching === FLOOR ? "This question only. Tap a re-ranker above to compare it with the search." : `This question only, the search against ${name(watching)}.`}
      </p>
      <div className="overflow-x-auto">
        <table className="numeric w-full border-collapse text-body">
          <thead>
            <tr>
              <th />
              {cols.map((m) => (
                <th key={m} scope="col" className="border-b-2 border-ink px-2 py-2 text-right font-serif font-semibold">
                  {name(m)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className="border-t border-line first:border-t-0">
                <td className="py-3 pr-2">
                  <span className="block font-serif">{r.name}</span>
                  <span className="mt-0.5 block text-micro text-ink-soft">{r.tech}</span>
                </td>
                {r.cells.map((c, i) => (
                  <td key={i} className={`px-2 py-3 text-right ${r.head ? "text-lead" : ""} ${typeof c === "string" && /[a-z]/.test(c) ? "font-serif text-micro text-ink-soft" : ""}`}>
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-micro text-ink-soft">Ranking quality comes from the run. The other rows are counted from each ranking&apos;s order.</p>
    </section>
  );
}
