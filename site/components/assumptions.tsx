import { Mono } from "@/components/chip";
import type { PerQuery } from "@/lib/per-query";
import { byMethod, pct, signed, type Results } from "@/lib/results";

function Boundary({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <li className="grid gap-1 border-l border-line pl-4">
      <h3 className="text-body font-semibold">{heading}</h3>
      <p className="max-w-[62ch] text-body leading-relaxed text-ink-soft">{children}</p>
    </li>
  );
}

/**
 * The boundaries of the result. These are not a disclaimer at the foot of the
 * page: each one is a thing a reader would otherwise assume, and each one changes
 * what the numbers above are allowed to mean.
 */
export function Assumptions({
  results,
  pilot,
  perQuery,
  records,
}: {
  results: Results;
  pilot: Results;
  perQuery: PerQuery;
  records: number;
}) {
  const jev = byMethod(results, "jev-score")!;
  const calls = jev.calls!;
  const laya = byMethod(results, "laya-score")!["ndcg@10_vs_bm25"]!;
  const pilotLaya = byMethod(pilot, "laya-score")!["ndcg@10_vs_bm25"]!;
  const ce = byMethod(results, "cross-encoder")!;
  const unjudged = perQuery.rows.filter((row) => row.relevant === 0).length;

  return (
    <ul className="grid gap-6">
      <Boundary heading={`On ${unjudged} of the ${perQuery.queries} queries, no re-ranker could have changed anything`}>
        BM25 retrieved {results.top_k} candidates for every query, and for {unjudged} of them not one of those{" "}
        {results.top_k} is judged relevant by the official qrels. nDCG@10 is zero for BM25 and zero for every
        re-ranker on those queries, whatever order they choose, so they sit on the zero line in the chart above
        and contribute nothing but denominator to every mean in the table. The result is therefore measured on
        the {perQuery.queries - unjudged} queries where re-ranking was possible at all, and reported over{" "}
        {perQuery.queries}.
      </Boundary>

      <Boundary heading={`${calls.failed} of ${calls.n.toLocaleString()} Jev calls never succeeded`}>
        {pct(calls.failed / calls.n, 2)} of them, mostly <Mono>HTTP 503</Mono>, after five retries with
        exponential backoff, and {calls.retries.toLocaleString()} retries across the run. Nothing was invented
        for a failed call: that passage kept the position BM25 gave it, marked <b>held</b> in the chart. Those
        held passages are BM25&apos;s judgement counted inside Jev&apos;s score.
      </Boundary>

      <Boundary heading={`Laya and the cross-encoder ran on ${results.device.toUpperCase()}`}>
        There was no usable GPU on the machine, so their latencies are {results.device.toUpperCase()} latencies
        and are not comparable to Laya&apos;s published 39.5 ms on a T4. Jev ran over the network, so its{" "}
        {jev.latency.p50_ms} ms median is a round trip and not a comparable quantity at all. Nothing on this
        page is a speed claim.
      </Boundary>

      <Boundary heading="The pilot and this run disagree about how much">
        A {pilot.queries}-query pilot at top-{pilot.top_k} put Laya {signed(pilotLaya.mean)} below the floor; at{" "}
        {results.queries} queries and top-{results.top_k} the loss is {signed(laya.mean)}. The direction held
        and the size did not, and the shallower candidate pool moved BM25&apos;s own floor with it. Where the
        two disagree, the number on this page is the one to believe, and the disagreement is the reason to
        distrust any single small run, including this one.
      </Boundary>

      <Boundary heading="Jev was never compared with the cross-encoder">
        Both beat BM25 with intervals clear of zero, and Jev is {signed(jev["ndcg@10"] - ce["ndcg@10"])}{" "}
        nDCG@10 ahead. That is not a result. Each interval is against the common floor, and two intervals that
        both exclude zero do not establish that one method beats the other. The paired Jev-minus-cross-encoder
        difference would, and it was not run.
      </Boundary>

      <Boundary
        heading={`${perQuery.rows.filter((row) => row.example).length} queries are browsable, ${perQuery.queries} are measured`}
      >
        The wire log is {records.toLocaleString()} records of roughly 3 KB, so the examples above are a curated
        subset chosen to span the outcomes, including the queries where Jev&apos;s calls failed. The table, the
        charts and the CSV are all {results.queries} queries. A subset picked for how well it reads would turn
        a negative result into a highlight reel, which is why the choosing is a tested function in the
        repository rather than a judgement made while writing the page.
      </Boundary>
    </ul>
  );
}
