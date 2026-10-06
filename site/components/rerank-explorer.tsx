"use client";

import { useMemo, useRef, useState } from "react";
import rerankData from "@/data/rerank.json";
import { Section } from "@/components/chip";
import { RerankDeltaBars } from "@/components/rerank-delta-bars";
import { RerankTheatre } from "@/components/rerank-theatre";
import { methodOrder, movableSplit, openingQuery, rerankerRows, type Rerank } from "@/lib/rerank";

const data = rerankData as Rerank;
const methods = methodOrder(data);
const rows = rerankerRows(data);
const split = movableSplit(data);

/**
 * The two interactive sections are one instrument, so they are one component:
 * the bars pick the query and the theatre shows what each method did to it. A
 * bar that only highlighted itself would be a chart; a bar that opens the
 * ranking it summarises is the way into 252 of them.
 *
 * The run file is imported here rather than handed down from the page, so it
 * travels once as a script chunk instead of twice, the second time inlined into
 * the HTML as props.
 */
export function RerankExplorer() {
  // Open where the best and the worst re-ranker disagree most, under the best:
  // switching to the worst then shows the finding on one question.
  const opening = useMemo(() => openingQuery(data, rows[0].method, rows[rows.length - 1].method), []);

  const [queryId, setQueryId] = useState(opening);
  const [method, setMethod] = useState(methods[1] ?? methods[0]);
  const theatre = useRef<HTMLDivElement>(null);

  const query = data.queries.find((q) => q.id === queryId) ?? data.queries[0];

  const fromBar = (id: string, barMethod: string) => {
    setQueryId(id);
    setMethod(barMethod);
    theatre.current?.scrollIntoView({
      block: "start",
      behavior:
        typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
    });
  };

  return (
    <>
      <div ref={theatre} className="scroll-mt-4">
        <Section
          id="theatre"
          title="Watch one question being re‑ranked"
          standfirst={`The first stage hands back ${data.summary.top_k} candidates in its own order. Each re-ranker scores every one of them and the list is sorted again. Choose a method and the same ${data.summary.top_k} documents travel to where it put them.`}
        >
          <RerankTheatre
            data={data}
            methods={methods}
            query={query}
            method={method}
            onSelectQuery={setQueryId}
            onSelectMethod={setMethod}
          />
        </Section>
      </div>

      <Section
        id="every-query"
        title="Every question at once"
        standfirst={`One bar per question, sorted by how far that method moved it against the ${data.floor} floor. Click a bar and that question opens above, under that method.`}
      >
        <RerankDeltaBars data={data} rows={rows} selected={queryId} onSelect={fromBar} />
        <p className="max-w-[72ch] text-body leading-relaxed text-ink-soft">
          {split.movable} of the {data.queries.length} questions are drawn. The other {split.excluded} have
          nothing relevant among their {data.summary.top_k} candidates, so every method scores zero on them
          whatever order it picks, and a bar of zero would say something the run does not.
        </p>
      </Section>
    </>
  );
}
