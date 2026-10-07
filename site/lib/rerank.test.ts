/**
 * The page states nothing it did not read out of the run, so the reading is what
 * is tested: the floor is never treated as a method, a delta is always against
 * that floor, the geometry puts zero exactly on the zero line, and a method that
 * has no recorded order for a query comes back as nothing rather than as the
 * floor's order wearing its name.
 *
 * The real run file is asserted against too, because a shape that holds for a
 * fixture and not for the run is not worth having.
 */
import { describe, expect, it } from "vitest";
import rerankData from "@/data/rerank.json";
import {
  openingQuery,
  verdict,
  label,
  matchQueries,
  methodOrder,
  movableSplit,
  queryNdcg,
  rerankerRows,
  runFacts,
  type Rerank,
} from "@/lib/rerank";

const run = rerankData as Rerank;

const fixture: Rerank = {
  floor: "hybrid",
  queries: [
    {
      id: "Q-1",
      text: "vitamin d",
      relevant: ["D-2"],
      can_move: true,
      ndcg: { "floor_ndcg@10": 0.5, "jev-score": 0.25, "cross-encoder": -0.1 },
    },
    {
      id: "Q-2",
      text: "deafness",
      relevant: [],
      can_move: false,
      ndcg: { "floor_ndcg@10": 0, "jev-score": 0, "cross-encoder": 0 },
    },
    {
      id: "Q-3",
      text: "vitamin c",
      relevant: ["D-1"],
      can_move: true,
      ndcg: { "floor_ndcg@10": 0.2, "jev-score": -0.05, "cross-encoder": 0.4 },
    },
  ],
  titles: { "D-1": "One", "D-2": "Two" },
  summary: {
    dataset: "fixture",
    queries: 3,
    top_k: 3,
    device: "cpu",
    commit: "abc1234",
    finished: "2026-01-01T00:00:00+00:00",
    documents: 40,
    head_to_head: [
      { method: "cross-encoder", against: "jev-score", mean: -0.05, ci95: [-0.08, -0.02], better: 1, worse: 2, same: 0 },
    ],
    methods: [
      {
        method: "hybrid",
        "ndcg@10": 0.3,
        "recall@10": 0.1,
        "mrr@10": 0.5,
        queries: 3,
        latency: { n: 0, p50_ms: null, p95_ms: null },
        scoring_wall_clock_s: 1,
      },
      {
        method: "cross-encoder",
        "ndcg@10": 0.31,
        "recall@10": 0.11,
        "mrr@10": 0.51,
        queries: 3,
        "ndcg@10_vs_floor": { mean: 0.01, ci95: [0.001, 0.02], better: 1, worse: 1, same: 1 },
        latency: { n: 9, p50_ms: 7, p95_ms: 11 },
        scoring_wall_clock_s: 2,
        calls: { n: 9, retries: 0, failed: 0, input_tokens: 0, market_cost_usd: 0 },
      },
      {
        method: "jev-score",
        "ndcg@10": 0.36,
        "recall@10": 0.12,
        "mrr@10": 0.55,
        queries: 3,
        "ndcg@10_vs_floor": { mean: 0.06, ci95: [0.03, 0.09], better: 2, worse: 1, same: 0 },
        latency: { n: 8, p50_ms: 300, p95_ms: 400 },
        scoring_wall_clock_s: 30,
        calls: { n: 9, retries: 0, failed: 1, input_tokens: 10, market_cost_usd: 0.25 },
      },
    ],
  },
  config: { top_k: 3, queries: 3, first_stage: "hybrid", encoder: "e", fusion: "reciprocal-rank", rrf_k: 60 },
};

describe("label", () => {
  it("never calls the floor BM25", () => {
    expect(label("hybrid").toLowerCase()).not.toContain("bm25");
    expect(label("hybrid")).toContain("hybrid");
  });

  it("falls back to the method's own name", () => {
    expect(label("something-new")).toBe("something-new");
  });
});

describe("methodOrder", () => {
  it("puts the floor first and the re-rankers best-first after it", () => {
    expect(methodOrder(fixture)).toEqual(["hybrid", "jev-score", "cross-encoder"]);
  });

  it("orders the real run the way the page reads it", () => {
    expect(methodOrder(run)).toEqual([
      "hybrid",
      "jev-score",
      "cross-encoder",
      "laya-score",
      "laya-typed-score",
    ]);
  });
});

describe("rerankerRows", () => {
  it("leaves the floor out, because the floor is what the others are measured against", () => {
    expect(rerankerRows(fixture).map((r) => r.method)).toEqual(["jev-score", "cross-encoder"]);
  });

  it("carries the mean and the interval through untouched", () => {
    const [best] = rerankerRows(fixture);
    expect(best.mean).toBe(0.06);
    expect(best.ci95).toEqual([0.03, 0.09]);
    expect(best.better).toBe(2);
  });

  it("marks an interval that stays on one side of zero", () => {
    expect(rerankerRows(fixture).every((r) => r.excludesZero)).toBe(true);
    expect(rerankerRows(run).every((r) => r.excludesZero)).toBe(true);
  });

  it("reads four re-rankers out of the run, two of them negative", () => {
    const rows = rerankerRows(run);
    expect(rows).toHaveLength(4);
    expect(rows.filter((r) => r.mean < 0).map((r) => r.method).sort()).toEqual([
      "laya-score",
      "laya-typed-score",
    ]);
  });
});

describe("runFacts", () => {
  it("adds the calls, the spend and the failures across every method", () => {
    const facts = runFacts(fixture);
    expect(facts.calls).toBe(18);
    expect(facts.spendUsd).toBeCloseTo(0.25, 10);
    expect(facts.failed).toBe(1);
  });

  it("carries the size of the corpus the questions were searched over", () => {
    expect(runFacts(fixture).documents).toBe(40);
  });

  it("reports the run's own queries, device and candidate depth", () => {
    const facts = runFacts(run);
    expect(facts.queries).toBe(run.summary.queries);
    expect(facts.device).toBe(run.summary.device);
    expect(facts.topK).toBe(run.summary.top_k);
    expect(facts.calls).toBe(
      run.summary.methods.reduce((total, m) => total + (m.calls?.n ?? 0), 0),
    );
  });
});

describe("movableSplit", () => {
  it("separates the queries with nothing relevant retrieved from the rest", () => {
    expect(movableSplit(fixture)).toEqual({ movable: 2, excluded: 1 });
  });

  it("finds the run's excluded queries", () => {
    const split = movableSplit(run);
    expect(split.movable + split.excluded).toBe(run.summary.queries);
    expect(split.excluded).toBeGreaterThan(0);
  });
});

describe("queryNdcg", () => {
  it("reads the floor's own score for the query", () => {
    expect(queryNdcg(fixture.queries[0], "hybrid", "hybrid")).toEqual({
      floor: 0.5,
      value: 0.5,
      delta: 0,
    });
  });

  it("adds the delta to the floor to get the method's score", () => {
    const at = queryNdcg(fixture.queries[0], "jev-score", "hybrid");
    expect(at.delta).toBe(0.25);
    expect(at.value).toBeCloseTo(0.75, 10);
  });

  it("says nothing rather than guessing when the method has no delta", () => {
    expect(queryNdcg(fixture.queries[0], "laya-score", "hybrid")).toEqual({
      floor: 0.5,
      value: null,
      delta: null,
    });
  });
});

describe("matchQueries", () => {
  it("returns everything for an empty search", () => {
    expect(matchQueries(fixture.queries, "  ")).toHaveLength(3);
  });

  it("matches the query text, ignoring case", () => {
    expect(matchQueries(fixture.queries, "VITAMIN").map((q) => q.id)).toEqual(["Q-1", "Q-3"]);
  });

  it("matches the query id too, because the ids are on screen", () => {
    expect(matchQueries(fixture.queries, "q-2").map((q) => q.id)).toEqual(["Q-2"]);
  });

  it("keeps the question on screen in the list even when the search excludes it", () => {
    expect(matchQueries(fixture.queries, "vitamin", "Q-2").map((q) => q.id)).toEqual(["Q-2", "Q-1", "Q-3"]);
  });

  it("does not list the question on screen twice", () => {
    expect(matchQueries(fixture.queries, "vitamin", "Q-1").map((q) => q.id)).toEqual(["Q-1", "Q-3"]);
  });
});

describe("verdict", () => {
  it("calls a method better only when its whole interval clears the floor", () => {
    expect(verdict(0.05, [0.03, 0.06])).toBe("better");
  });

  it("calls it worse when the whole interval sits below", () => {
    expect(verdict(-0.02, [-0.04, -0.01])).toBe("worse");
  });

  it("refuses to call an interval that crosses zero either way", () => {
    expect(verdict(0.01, [-0.01, 0.03])).toBe("no clear change");
    expect(verdict(0.01, null)).toBe("no clear change");
  });
});

describe("openingQuery", () => {
  it("opens where one method helped and the other hurt, by the widest margin", () => {
    expect(openingQuery(fixture, "jev-score", "cross-encoder")).toBe("Q-1");
    expect(openingQuery(fixture, "cross-encoder", "jev-score")).toBe("Q-3");
  });

  it("never opens on a question nothing could move", () => {
    const id = openingQuery(run, "jev-score", "laya-score");
    const query = run.queries.find((q) => q.id === id)!;
    expect(query.can_move).toBe(true);
    expect(query.ndcg["jev-score"]).toBeGreaterThan(0);
    expect(query.ndcg["laya-score"]).toBeLessThan(0);
  });
});
