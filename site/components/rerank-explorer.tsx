"use client";

import { useState } from "react";
import rerankData from "@/data/rerank.json";
import { Section } from "@/components/chip";
import { RerankWatch } from "@/components/rerank-watch";
import { methodOrder, type Rerank } from "@/lib/rerank";
import { storyQuestionId } from "@/lib/rerank-story";

const data = rerankData as Rerank;
const methods = methodOrder(data);
// The question the story at the top of the page follows, so the reader arrives
// here already knowing it.
const opening = storyQuestionId(data);
const openingText = data.queries.find((q) => q.id === opening)?.text ?? "";

/**
 * See it for yourself: every question, every re-ranker, opening on the one the
 * story told.
 *
 * The run file is imported here rather than handed down from the page, so it
 * travels once as a script chunk instead of twice, the second time inlined into
 * the HTML as props.
 */
export function RerankExplorer() {
  const [queryId, setQueryId] = useState(opening);
  const [method, setMethod] = useState(methods[1] ?? methods[0]);

  return (
    <Section
      id="theatre"
      title="See it for yourself."
      standfirst={`Everything above came from averages and one example. Here are all ${data.queries.length} questions. It opens on “${openingText}”, the question from the top. Pick any other and watch each re-ranker reorder its ${data.summary.top_k}.`}
    >
      <RerankWatch queryId={queryId} onQuery={setQueryId} watching={method} onWatch={setMethod} />
    </Section>
  );
}
