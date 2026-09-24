"use client";

import { useEffect, useState } from "react";
import { Code, CopyButton } from "@/components/code";
import { Chip, Mono } from "@/components/chip";
import { columns, reasonLabel, type Example, type ExamplesIndex, type Line } from "@/lib/example";
import { label, signed } from "@/lib/results";

const METHODS = ["jev-score", "cross-encoder", "laya-score", "laya-typed-score"];

// The picker is grouped by why each query is in the subset, so the spread of
// outcomes is visible before anything is clicked.
const GROUPS = ["jev-gain", "laya-loss", "unchanged", "jev-failure"];

/**
 * One query, worked through. BM25 keeps the left-hand column on every view,
 * because it is the floor the whole experiment is measured against; the picker
 * only changes what sits beside it.
 */
export function WorkedExample({ index }: { index: ExamplesIndex }) {
  const ordered = GROUPS.flatMap((g) => index.exported.filter((e) => e.reason === g));
  const [queryId, setQueryId] = useState(ordered[0].query_id);
  const [method, setMethod] = useState(METHODS[0]);
  // The fetch result carries the id it was fetched for, and the opened passage
  // carries the query and method it belongs to, so switching either one clears
  // what is on screen by itself rather than through a second render.
  const [loaded, setLoaded] = useState<{ id: string; example?: Example; error?: string } | null>(null);
  const [opened, setOpened] = useState<{ id: string; method: string; docId: string } | null>(null);

  useEffect(() => {
    let live = true;
    fetch(`/examples/${queryId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data: Example) => live && setLoaded({ id: queryId, example: data }))
      .catch((e: Error) => live && setLoaded({ id: queryId, error: e.message }));
    return () => {
      live = false;
    };
  }, [queryId]);

  const example = loaded?.id === queryId ? loaded.example ?? null : null;
  const error = loaded?.id === queryId ? loaded.error ?? null : null;
  const open = opened && opened.id === queryId && opened.method === method ? opened.docId : null;
  const setOpen = (docId: string | null) =>
    setOpened(docId === null ? null : { id: queryId, method, docId });

  const entry = index.exported.find((e) => e.query_id === queryId)!;
  const { bm25, method: reordered } = example
    ? columns(example, method)
    : { bm25: [] as Line[], method: [] as Line[] };
  const record = example && open ? example.wire[method]?.[open] : null;

  return (
    <div className="grid gap-5">
      <div className="grid gap-4 border border-line bg-surface p-4 md:p-5">
        <div className="grid gap-2 sm:max-w-[36rem]">
          <label htmlFor="query-picker" className="text-micro font-semibold text-ink-soft">
            Query — {index.exported.length} of {index.queries_in_run} in the run
          </label>
          <select
            id="query-picker"
            value={queryId}
            onChange={(e) => setQueryId(e.target.value)}
            className="w-full border border-line-strong bg-page px-3 py-2 text-body"
          >
            {GROUPS.map((group) => {
              const entries = ordered.filter((e) => e.reason === group);
              if (!entries.length) return null;
              return (
                <optgroup key={group} label={reasonLabel(group)}>
                  {entries.map((e) => (
                    <option key={e.query_id} value={e.query_id}>{e.query}</option>
                  ))}
                </optgroup>
              );
            })}
          </select>
        </div>

        <fieldset className="grid gap-2">
          <legend className="text-micro font-semibold text-ink-soft">
            Compare BM25 against
          </legend>
          <div className="flex flex-wrap gap-2">
            {METHODS.map((m) => (
              <button
                key={m}
                type="button"
                aria-pressed={m === method}
                onClick={() => setMethod(m)}
                className={`border px-3 py-1.5 text-micro font-semibold ${
                  m === method
                    ? "border-accent bg-accent text-accent-ink"
                    : "border-line-strong bg-page text-ink-soft hover:text-ink"
                }`}
              >
                {label(m)}
              </button>
            ))}
          </div>
        </fieldset>

        <p className="numeric max-w-[70ch] text-micro text-ink-soft">
          {entry.candidates} candidates, {entry.relevant} of them judged relevant.
          {" "}In this example: {reasonLabel(entry.reason)}.
          {entry.jev_failed > 0 && ` ${entry.jev_failed} Jev calls never came back.`}
          {example && (
            <>
              {" "}
              <b className={example.methods[method].delta >= 0 ? "text-clear" : "text-danger"}>
                {signed(example.methods[method].delta)} nDCG@10
              </b>{" "}
              against BM25 on this query.
            </>
          )}
        </p>
      </div>

      {error && (
        <p role="status" className="border border-danger bg-surface p-4 text-body text-danger">
          Could not load {queryId}: {error}
        </p>
      )}
      {!example && !error && (
        <p role="status" className="text-body text-ink-soft">Loading {queryId}…</p>
      )}

      {example && (
        <>
          <p className="max-w-[70ch] text-lead font-semibold">“{example.query}”</p>
          <div className="grid gap-5 md:grid-cols-2">
            <Column
              heading="BM25"
              note="the fixed reference"
              lines={bm25}
              open={open}
              onOpen={() => {}}
              interactive={false}
            />
            <Column
              heading={label(method)}
              note="re-ordered, click a passage for the wire"
              lines={reordered}
              open={open}
              onOpen={setOpen}
              interactive
            />
          </div>
          <Legend />

          {record && (
            <div className="grid gap-3 border border-line bg-surface p-4 md:p-5">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <h3 className="text-body font-extrabold">
                  <Mono>{open}</Mono> — the exact call
                  {record.failed && <Chip>failed</Chip>}
                  {typeof record.latency_ms === "number" && record.latency_ms > 0 && (
                    <Chip>{record.latency_ms.toFixed(0)} ms</Chip>
                  )}
                  {typeof record.retries === "number" && record.retries > 0 && (
                    <Chip>{record.retries} retries</Chip>
                  )}
                </h3>
                <button
                  type="button"
                  onClick={() => setOpen(null)}
                  className="text-micro font-semibold text-ink-soft hover:text-ink"
                >
                  Close
                </button>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="grid min-w-0 gap-2">
                  <div className="flex items-baseline justify-between gap-3">
                    <h4 className="text-micro font-semibold text-ink-soft">Request sent</h4>
                    <CopyButton value={record.request} />
                  </div>
                  <Code value={record.request} />
                </div>
                <div className="grid min-w-0 gap-2">
                  <div className="flex items-baseline justify-between gap-3">
                    <h4 className="text-micro font-semibold text-ink-soft">Response received</h4>
                    <CopyButton value={record.response} />
                  </div>
                  <Code value={record.response} />
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Column({ heading, note, lines, open, onOpen, interactive }: {
  heading: string;
  note: string;
  lines: Line[];
  open: string | null;
  onOpen: (docId: string) => void;
  interactive: boolean;
}) {
  return (
    <section className="grid min-w-0 gap-2" aria-label={`${heading} ranking`}>
      <h3 className="text-body font-extrabold">
        {heading} <span className="font-normal text-ink-soft">— {note}</span>
      </h3>
      <ol className="grid min-w-0 gap-px border border-line bg-line">
        {lines.map((line) => (
          <li key={line.docId} className="min-w-0 bg-surface">
            <Row line={line} open={open === line.docId} onOpen={onOpen} interactive={interactive} />
          </li>
        ))}
      </ol>
    </section>
  );
}

function Row({ line, open, onOpen, interactive }: {
  line: Line; open: boolean; onOpen: (docId: string) => void; interactive: boolean;
}) {
  const body = (
    <>
      <span className="numeric w-6 shrink-0 text-micro font-extrabold text-ink-soft">{line.position}</span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-micro font-semibold">
          {line.relevant && (
            <span className="mr-1.5 font-extrabold text-accent" title={`judged relevant, grade ${line.grade}`}>
              ●<span className="sr-only">judged relevant, grade {line.grade} — </span>
            </span>
          )}
          {line.passage?.title || line.docId}
        </span>
        <span className="numeric block truncate font-mono text-micro text-ink-soft">
          {line.docId}
          {line.score !== null && ` · ${line.score.toFixed(4)}`}
          {line.failed && " · call failed, held its BM25 slot"}
        </span>
      </span>
      {interactive && <Move move={line.move} failed={line.failed} />}
    </>
  );
  const shell = "flex w-full min-w-0 items-center gap-2 px-2 py-2 text-left";
  if (!interactive) return <div className={shell}>{body}</div>;
  return (
    <button
      type="button"
      onClick={() => onOpen(line.docId)}
      aria-expanded={open}
      className={`${shell} ${open ? "bg-accent-wash" : "hover:bg-sunk"}`}
    >
      {body}
    </button>
  );
}

function Move({ move, failed }: { move: number; failed: boolean }) {
  if (move === 0) {
    return (
      <span className="numeric w-12 shrink-0 text-right text-micro text-ink-soft">
        {failed ? "held" : "—"}
        <span className="sr-only"> unchanged</span>
      </span>
    );
  }
  return (
    <span
      className={`numeric w-12 shrink-0 text-right text-micro font-extrabold ${
        move > 0 ? "text-clear" : "text-danger"
      }`}
    >
      {move > 0 ? "▲" : "▼"} {Math.abs(move)}
      <span className="sr-only"> places {move > 0 ? "up" : "down"}</span>
    </span>
  );
}

function Legend() {
  return (
    <p className="max-w-[70ch] text-micro leading-relaxed text-ink-soft">
      <span className="font-extrabold text-accent">●</span> marks a passage the official qrels judge relevant to
      this query. <b className="text-clear">▲</b> and <b className="text-danger">▼</b> are places moved against
      BM25&apos;s own ordering of the same candidates. A passage whose call failed is marked{" "}
      <b>held</b>: nothing was invented for it, so it keeps the slot BM25 gave it.
    </p>
  );
}
