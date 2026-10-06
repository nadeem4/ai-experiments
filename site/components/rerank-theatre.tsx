"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { RerankQueryPicker } from "@/components/rerank-query-picker";
import {
  candidateRows,
  label,
  queryNdcg,
  runFacts,
  scoreNote,
  signed,
  type Detail,
  type Rerank,
  type RerankQuery,
} from "@/lib/rerank";

const DETAIL_URL = "/rerank-detail.json";

/**
 * The re-ranking itself, one question at a time.
 *
 * The tables elsewhere on this page are averages, and an average cannot show the
 * thing that actually happens when a re-ranker runs: a document that the
 * retriever had at rank fourteen arriving at rank two. So the twenty candidates
 * are a single list that is never rebuilt, only reordered, and choosing a method
 * animates each row from where it was to where that method put it. Under
 * `prefers-reduced-motion` the rows cross-fade in place instead, which says the
 * same thing without moving anything.
 *
 * The detail file is 1.4MB, so it is fetched when the section comes into view
 * rather than imported into the page.
 */
export function RerankTheatre({
  data,
  methods,
  query,
  method,
  onSelectQuery,
  onSelectMethod,
}: {
  data: Rerank;
  methods: string[];
  query: RerankQuery;
  method: string;
  onSelectQuery: (id: string) => void;
  onSelectMethod: (method: string) => void;
}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const holder = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    let alive = true;
    const load = () => {
      fetch(DETAIL_URL)
        .then((response) => (response.ok ? response.json() : Promise.reject(new Error("unavailable"))))
        .then((body: Detail) => alive && setDetail(body))
        .catch(() => alive && setFailed(true));
    };
    if (typeof IntersectionObserver === "undefined") {
      load();
      return () => {
        alive = false;
      };
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          load();
        }
      },
      { rootMargin: "600px" },
    );
    observer.observe(node);
    return () => {
      alive = false;
      observer.disconnect();
    };
  }, []);

  const rows = useMemo(
    () =>
      detail
        ? candidateRows(detail.queries[query.id], method, data.floor, query.relevant, data.titles)
        : null,
    [detail, query, method, data],
  );

  const at = queryNdcg(query, method, data.floor);
  const facts = runFacts(data);

  return (
    <div ref={holder} className="grid min-w-0 gap-5">
      <RerankQueryPicker queries={data.queries} selected={query.id} onSelect={onSelectQuery} />

      <div className="grid min-w-0 gap-1">
        <p className="numeric text-micro text-ink-soft">
          {query.id} · {query.relevant.length} of {facts.topK} candidates judged relevant
          {query.can_move ? "" : " · nothing relevant was retrieved, so no order can score above zero"}
        </p>
        <p className="text-lead leading-snug">{query.text}</p>
      </div>

      <div className="flex min-w-0 flex-wrap gap-2" role="group" aria-label="Ranking to show">
        {methods.map((name) => {
          const active = name === method;
          return (
            <button
              key={name}
              type="button"
              aria-pressed={active}
              onClick={() => onSelectMethod(name)}
              className={`numeric min-h-11 border px-3 py-1.5 text-micro ${
                active
                  ? "border-line-strong bg-sunk font-medium text-ink"
                  : "border-line bg-surface text-ink-soft hover:text-ink"
              }`}
            >
              {name === data.floor ? "the floor" : label(name)}
            </button>
          );
        })}
      </div>

      <p className="numeric max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        {method === data.floor ? (
          <>
            nDCG@10 {at.floor.toFixed(4)} — the order the retriever returned, and the line every other
            method on this question is measured from.
          </>
        ) : at.value === null ? (
          <>nDCG@10 for {label(method)} on this question was not recorded.</>
        ) : (
          <>
            nDCG@10 {at.value.toFixed(4)} against the floor&apos;s {at.floor.toFixed(4)}:{" "}
            <b style={{ color: at.delta! > 0 ? "var(--up)" : at.delta! < 0 ? "var(--down)" : "var(--ink)" }}>
              {signed(at.delta!)}
            </b>{" "}
            on this question.
          </>
        )}
      </p>

      {rows ? (
        <CandidateList rows={rows} method={method} floor={data.floor} query={query} open={open} setOpen={setOpen} detail={detail!} />
      ) : failed ? (
        <Notice>
          The candidate file did not load, so there is nothing to reorder here. Every number above still comes
          from the run summary.
        </Notice>
      ) : !detail ? (
        <Notice>Loading the {facts.topK} candidates…</Notice>
      ) : (
        <Notice>
          The run holds no <span className="numeric">{method}</span> order for{" "}
          <span className="numeric">{query.id}</span>, so there is nothing to show. It is not the floor&apos;s
          order under another name: {facts.failed} of the run&apos;s {facts.calls.toLocaleString()} calls
          failed, and nothing was invented in place of one.
        </Notice>
      )}
    </div>
  );
}

function Notice({ children }: { children: React.ReactNode }) {
  return (
    <p className="max-w-[72ch] border border-line bg-surface p-4 text-micro leading-relaxed text-ink-soft">
      {children}
    </p>
  );
}

function CandidateList({
  rows,
  method,
  floor,
  query,
  open,
  setOpen,
  detail,
}: {
  rows: NonNullable<ReturnType<typeof candidateRows>>;
  method: string;
  floor: string;
  query: RerankQuery;
  open: string | null;
  setOpen: (id: string | null) => void;
  detail: Detail;
}) {
  const nodes = useRef(new Map<string, HTMLLIElement>());
  const tops = useRef(new Map<string, number>());

  // FLIP. The previous layout was measured at the end of the last render, so a
  // row that has moved can be put back where it was and then released.
  useLayoutEffect(() => {
    const reduce =
      typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const next = new Map<string, number>();
    nodes.current.forEach((node, id) => next.set(id, node.offsetTop));
    nodes.current.forEach((node, id) => {
      const from = tops.current.get(id);
      const to = next.get(id)!;
      if (from === undefined || from === to) return;
      if (reduce) {
        node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 200, easing: "ease-out" });
      } else {
        node.animate(
          [{ transform: `translateY(${from - to}px)` }, { transform: "translateY(0)" }],
          { duration: 460, easing: "cubic-bezier(0.2, 0.85, 0.25, 1)" },
        );
      }
    });
    tops.current = next;
  }, [rows, open]);

  return (
    <ol className="relative grid min-w-0 gap-px border border-line bg-line">
      {rows.map((row) => {
        const expanded = open === row.docId;
        const travel = method === floor ? 0 : row.moved;
        return (
          <li
            key={row.docId}
            ref={(node) => {
              if (node) nodes.current.set(row.docId, node);
              else nodes.current.delete(row.docId);
            }}
            className="min-w-0 bg-surface"
          >
            <button
              type="button"
              aria-expanded={expanded}
              onClick={() => setOpen(expanded ? null : row.docId)}
              className="flex w-full min-w-0 items-baseline gap-3 px-3 py-2 text-left"
            >
              <span className="numeric w-6 shrink-0 text-micro text-ink-soft">
                {String(row.rank).padStart(2, "0")}
              </span>
              {/* On a phone the id column took a quarter of the row and squeezed every
                  title to six lines; the id is still in the opened row. */}
              <span
                className="numeric hidden w-[5.5rem] shrink-0 text-micro sm:inline"
                style={{ color: row.relevant ? "var(--mark)" : "var(--ink-soft)" }}
              >
                {row.docId}
              </span>
              <span
                className="min-w-0 flex-1 text-micro leading-snug"
                style={{
                  color: row.relevant ? "var(--mark)" : "var(--ink)",
                  fontWeight: row.relevant ? 500 : 400,
                }}
              >
                {row.title}
                {row.relevant && (
                  <span className="numeric ml-2 whitespace-nowrap border border-mark px-1 align-baseline text-micro font-normal">
                    relevant
                  </span>
                )}
              </span>
              {/* Where it came from rather than how far it went: "from 14" next to a
                  row at 02 says both, and needs no arrow to decode. */}
              <span
                className="numeric w-16 shrink-0 text-right text-micro"
                style={{
                  color: travel > 0 ? "var(--up)" : travel < 0 ? "var(--down)" : "var(--ink-soft)",
                }}
                aria-label={travel === 0 ? "unmoved" : `moved ${travel > 0 ? "up" : "down"} from ${row.floorRank}`}
              >
                {travel === 0 ? "same" : `from ${String(row.floorRank).padStart(2, "0")}`}
              </span>
            </button>
            {expanded && (
              <div className="border-t border-line bg-page px-3 py-3 pl-12">
                <p className="numeric text-micro text-ink-soft">
                  {row.docId} · {scoreNote(method, floor, row.score)}
                  {method !== floor && ` · the floor had it at ${String(row.floorRank).padStart(2, "0")}`}
                </p>
                <p className="mt-2 max-w-[72ch] text-micro leading-relaxed text-ink-soft">
                  {snippet(detail.passages[row.docId])}
                </p>
              </div>
            )}
          </li>
        );
      })}
      <li className="bg-surface px-3 py-2">
        <p className="numeric text-micro text-ink-soft">
          {query.relevant.length
            ? "Documents the official judgements call relevant are tagged and set in violet."
            : "Nothing among these candidates is judged relevant to this question."}{" "}
          Open a row for the passage and the score behind it.
        </p>
      </li>
    </ol>
  );
}

/** Enough of the abstract to see why a passage was or was not ranked here. */
function snippet(passage: string | undefined, limit = 420): string {
  if (!passage) return "The run kept no passage text for this document.";
  return passage.length <= limit ? passage : `${passage.slice(0, limit).trimEnd()}…`;
}
