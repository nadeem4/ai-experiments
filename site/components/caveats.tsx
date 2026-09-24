import { Mono } from "@/components/chip";
import { byMethod, pct, signed, type Results } from "@/lib/results";

/** What the run does not establish, from the experiment's own honest section. */
export function Caveats({ results, pilot }: { results: Results; pilot: Results }) {
  const jev = byMethod(results, "jev-score")!;
  const calls = jev.calls!;
  const laya = byMethod(results, "laya-score")!["ndcg@10_vs_bm25"]!;
  const pilotLaya = byMethod(pilot, "laya-score")!["ndcg@10_vs_bm25"]!;
  const ce = byMethod(results, "cross-encoder")!;

  return (
    <ul className="grid max-w-[70ch] gap-5">
      <li className="border-l-2 border-line pl-4 text-body leading-relaxed">
        <b>{calls.failed} of {calls.n.toLocaleString()} Jev calls never succeeded</b> — {pct(calls.failed / calls.n, 2)},
        mostly <Mono>HTTP 503</Mono>, after five retries with exponential backoff. Nothing was invented for
        them: those passages kept the position BM25 gave them, which you can see happen on the queries marked
        as failures in the picker above. There were {calls.retries.toLocaleString()} retries across the run.
      </li>
      <li className="border-l-2 border-line pl-4 text-body leading-relaxed">
        <b>Laya and the cross-encoder ran on {results.device.toUpperCase()}.</b> There was no usable GPU on the
        machine, so their latencies are {results.device.toUpperCase()} latencies and are not comparable to
        Laya&apos;s published 39.5 ms on a T4. Jev ran over the network, so its {jev.latency.p50_ms} ms median
        is a round trip and not a comparable quantity at all.
      </li>
      <li className="border-l-2 border-line pl-4 text-body leading-relaxed">
        <b>The pilot and this run disagree in magnitude.</b> A {pilot.queries}-query pilot at top-{pilot.top_k}{" "}
        put Laya {signed(pilotLaya.mean)} below the floor; at {results.queries} queries and top-{results.top_k}{" "}
        the loss is {signed(laya.mean)}. The direction held, the size did not, and the shallower candidate pool
        moved BM25&apos;s own floor too. Where they disagree, the number on this page is the one to believe.
      </li>
      <li className="border-l-2 border-line pl-4 text-body leading-relaxed">
        <b>Jev leading the cross-encoder was never tested directly.</b> Both beat BM25 with intervals clear of
        zero, and Jev is {signed(jev["ndcg@10"] - ce["ndcg@10"])} nDCG@10 ahead, but each interval is against
        the common floor. Two intervals that both exclude zero do not establish that one method beats the
        other; the paired Jev-minus-cross-encoder difference would, and it was not run.
      </li>
    </ul>
  );
}
