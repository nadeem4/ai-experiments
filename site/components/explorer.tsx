"use client";

import { useEffect, useMemo, useState } from "react";
import { Chip, Mono } from "@/components/chip";
import { Code, CopyButton } from "@/components/code";
import { SlopeChart } from "@/components/slope-chart";
import { StripPlot } from "@/components/strip-plot";
import type { Example } from "@/lib/example";
import type { PerQuery } from "@/lib/per-query";
import { label, signed } from "@/lib/results";
import { slopeLines } from "@/lib/slope";

/**
 * The strip plot and the worked example are one thing, not two: the chart picks
 * the query and the method, and the slope chart below shows what that choice did
 * to one ranking. There is no separate query picker, because a reader choosing a
 * query from a dropdown is choosing a name, and a reader choosing a dot is
 * choosing a result.
 */
export function Explorer({
  perQuery,
  methods,
  means,
  records,
}: {
  perQuery: PerQuery;
  methods: string[];
  means: Record<string, number>;
  records: number;
}) {
  // The richest example to open on: most passages judged relevant, so the first
  // thing on screen has something for the re-ranker to have moved.
  const first = useMemo(
    () =>
      perQuery.rows
        .filter((row) => row.example)
        .reduce((best, row) => (row.relevant > best.relevant ? row : best)).id,
    [perQuery],
  );

  const [queryId, setQueryId] = useState(first);
  const [method, setMethod] = useState(methods[0]);
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

  const example = loaded?.id === queryId ? (loaded.example ?? null) : null;
  const error = loaded?.id === queryId ? (loaded.error ?? null) : null;
  const open = opened && opened.id === queryId && opened.method === method ? opened.docId : null;
  const row = perQuery.rows.find((r) => r.id === queryId)!;
  const ordering = example?.methods[method];
  const lines = example ? slopeLines(example.bm25, ordering?.order ?? [], example.relevant) : [];
  const record = example && open ? example.wire[method]?.[open] : null;

  return (
    <div className="grid min-w-0 gap-6">
      <fieldset className="grid gap-2">
        <legend className="mb-2 text-body">Which re-ranker to read</legend>
        <div className="flex flex-wrap gap-2">
          {methods.map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={m === method}
              onClick={() => setMethod(m)}
              className={`numeric border px-3 py-1.5 text-micro ${
                m === method
                  ? "border-ink bg-ink text-page"
                  : "border-line-strong bg-surface text-ink-soft hover:text-ink"
              }`}
            >
              {label(m)}
            </button>
          ))}
        </div>
      </fieldset>

      <StripPlot
        data={perQuery}
        methods={methods}
        active={method}
        selected={queryId}
        onSelect={setQueryId}
        means={means}
      />

      <p className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        One dot is one query. Dots right of the line are queries the re-ranker improved, dots left of it are
        queries it damaged, and the column standing on the line is every query whose nDCG@10 did not move.
        Ringed dots are the {perQuery.rows.filter((r) => r.example).length} queries whose full wire was
        exported; in the highlighted row they open below. The other {perQuery.queries - perQuery.rows.filter((r) => r.example).length}{" "}
        dots are readings, not links: the run recorded {records.toLocaleString()} calls of roughly 3 KB, and a
        browser cannot be sent all of them.
      </p>

      <div className="border-t border-line pt-5">
        {error && (
          <p role="status" className="border border-down bg-surface p-4 text-body text-down">
            Could not load {queryId}. {error}. Reload the page, or open the file at{" "}
            <Mono>/examples/{queryId}.json</Mono>.
          </p>
        )}
        {!example && !error && (
          <p role="status" className="text-body text-ink-soft">
            Loading {queryId}.
          </p>
        )}

        {example && (
          <div className="grid min-w-0 gap-5">
            <div className="grid gap-2">
              <h3 className="max-w-[46ch] text-lead font-semibold italic leading-snug">
                {example.query}
              </h3>
              <p className="numeric text-micro text-ink-soft">
                {example.query_id} · {example.bm25.length} candidates · {row.relevant} judged relevant ·
                BM25 scored {row.bm25.toFixed(4)} ·{" "}
                <b style={{ color: row.deltas[method] >= 0 ? "var(--up)" : "var(--down)" }}>
                  {signed(row.deltas[method])} with {label(method)}
                </b>
              </p>
            </div>

            <SlopeChart
              key={queryId}
              lines={lines}
              passages={example.passages}
              scores={ordering?.scores ?? {}}
              failed={ordering?.failed ?? []}
              methodLabel={label(method)}
              open={open}
              onOpen={(docId) => setOpened({ id: queryId, method, docId })}
            />

            <p className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
              A filled violet square marks a passage the official qrels judge relevant to this query. A line
              rising to the right is a passage the re-ranker promoted, a falling line is one it demoted, and a
              flat grey line is one it left alone. Pick any passage on the right for the exact request and
              response behind it. <b>held</b> marks a call that never came back after five retries: nothing was
              invented for it, so it kept the slot BM25 gave it.
            </p>

            {record && (
              <div className="grid gap-3 border border-line bg-surface p-4 md:p-5">
                <div className="flex flex-wrap items-baseline justify-between gap-3">
                  <h4 className="text-body font-semibold">
                    <Mono>{open}</Mono>, the exact call
                    {record.failed && <Chip>failed</Chip>}
                    {typeof record.latency_ms === "number" && record.latency_ms > 0 && (
                      <Chip>{record.latency_ms.toFixed(0)} ms</Chip>
                    )}
                    {typeof record.retries === "number" && record.retries > 0 && (
                      <Chip>{record.retries} retries</Chip>
                    )}
                  </h4>
                  <button
                    type="button"
                    onClick={() => setOpened(null)}
                    className="numeric text-micro text-ink-soft hover:text-ink"
                  >
                    Close
                  </button>
                </div>
                <div className="grid gap-4 lg:grid-cols-2">
                  <div className="grid min-w-0 gap-2">
                    <div className="flex items-baseline justify-between gap-3">
                      <h5 className="numeric text-micro text-ink-soft">Request sent</h5>
                      <CopyButton value={record.request} />
                    </div>
                    <Code value={record.request} />
                  </div>
                  <div className="grid min-w-0 gap-2">
                    <div className="flex items-baseline justify-between gap-3">
                      <h5 className="numeric text-micro text-ink-soft">Response received</h5>
                      <CopyButton value={record.response} />
                    </div>
                    <Code value={record.response} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
