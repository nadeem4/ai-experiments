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
  barIndexAt,
  barLayout,
  barPaths,
  candidateRows,
  scoreNote,
  deltaSeries,
  axis,
  label,
  matchQueries,
  methodOrder,
  movableSplit,
  queryNdcg,
  rerankerRows,
  runFacts,
  symmetricHalf,
  type Detail,
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

const detail: Detail = {
  queries: {
    "Q-1": {
      orders: {
        hybrid: ["D-1", "D-2", "D-3"],
        "jev-score": ["D-2", "D-3", "D-1"],
      },
      scores: { "jev-score": [3.5, 1.25, 0.5] },
    },
  },
  passages: { "D-1": "first passage", "D-2": "second passage" },
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

describe("symmetricHalf", () => {
  it("has a domain even when there is nothing to plot", () => {
    expect(symmetricHalf([], 0.01)).toBeCloseTo(0.01, 10);
  });

  it("covers the largest value in either direction, rounded out to a readable tick", () => {
    expect(symmetricHalf([0.031, -0.02], 0.01)).toBeCloseTo(0.04, 10);
    expect(symmetricHalf([-0.62, 0.1], 0.25)).toBeCloseTo(0.75, 10);
  });
});

describe("axis", () => {
  it("puts zero in the middle and a gain to the right of it", () => {
    const a = axis([0.05, -0.05], 1000, 0.01);
    expect(a.x(0)).toBe(500);
    expect(a.x(0.05)).toBeGreaterThan(500);
    expect(a.x(-0.05)).toBe(1000 - a.x(0.05));
  });

  it("offers ticks that include zero and both ends", () => {
    const a = axis([0.05], 1000, 0.01);
    expect(a.ticks[0]).toBeCloseTo(-a.half, 10);
    expect(a.ticks).toContain(0);
    expect(a.ticks[a.ticks.length - 1]).toBeCloseTo(a.half, 10);
  });
});

describe("deltaSeries", () => {
  it("keeps only the queries that could move", () => {
    expect(deltaSeries(fixture, "jev-score").map((d) => d.id)).toEqual(["Q-1", "Q-3"]);
  });

  it("sorts by the delta, largest gain first", () => {
    expect(deltaSeries(fixture, "cross-encoder").map((d) => d.delta)).toEqual([0.4, -0.1]);
  });

  it("carries the query text so a bar can name itself", () => {
    expect(deltaSeries(fixture, "jev-score")[0].text).toBe("vitamin d");
  });

  it("covers every movable query of the run for every re-ranker", () => {
    const { movable } = movableSplit(run);
    for (const row of rerankerRows(run)) {
      expect(deltaSeries(run, row.method)).toHaveLength(movable);
    }
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

describe("barLayout", () => {
  const series = [
    { id: "a", text: "a", delta: 0.4 },
    { id: "b", text: "b", delta: 0 },
    { id: "c", text: "c", delta: -0.2 },
  ];

  it("hangs a gain above the zero line and a loss below it", () => {
    const { bars, zeroY } = barLayout(series, { width: 300, height: 100, step: 0.1 });
    expect(bars[0].y + bars[0].height).toBeCloseTo(zeroY, 10);
    expect(bars[0].y).toBeLessThan(zeroY);
    expect(bars[2].y).toBeCloseTo(zeroY, 10);
    expect(bars[2].height).toBeGreaterThan(0);
  });

  it("gives a query that did not move no height at all", () => {
    const { bars } = barLayout(series, { width: 300, height: 100, step: 0.1 });
    expect(bars[1].height).toBe(0);
  });

  it("lays the bars left to right in the order it was given", () => {
    const { bars } = barLayout(series, { width: 300, height: 100, step: 0.1 });
    expect(bars.map((b) => b.id)).toEqual(["a", "b", "c"]);
    expect(bars[0].x).toBeLessThan(bars[1].x);
    expect(bars[2].x + bars[2].width).toBeLessThanOrEqual(300);
  });

  it("keeps every bar wide enough to see and to hit", () => {
    const many = Array.from({ length: 252 }, (_, i) => ({ id: `q${i}`, text: "", delta: 0.1 }));
    const { bars } = barLayout(many, { width: 1000, height: 100, step: 0.1 });
    expect(bars).toHaveLength(252);
    expect(bars.every((b) => b.width > 0)).toBe(true);
  });
});

describe("barPaths", () => {
  const series = [
    { id: "a", text: "a", delta: 0.4 },
    { id: "b", text: "b", delta: 0 },
    { id: "c", text: "c", delta: -0.2 },
  ];
  const { bars, zeroY } = barLayout(series, { width: 300, height: 100, step: 0.1 });

  it("sorts the bars into a gain path, a loss path and a path of the unmoved", () => {
    const paths = barPaths(bars);
    expect(paths.gain.match(/M/g)).toHaveLength(1);
    expect(paths.loss.match(/M/g)).toHaveLength(1);
    expect(paths.flat.match(/M/g)).toHaveLength(1);
  });

  it("starts a gain above the zero line and a loss on it", () => {
    const paths = barPaths(bars);
    expect(Number(paths.gain.slice(1).split(" ")[1].split("h")[0])).toBeLessThan(zeroY);
    expect(Number(paths.loss.slice(1).split(" ")[1].split("h")[0])).toBeCloseTo(zeroY, 1);
  });

  it("leaves a path empty when nothing went that way", () => {
    const only = barLayout([{ id: "a", text: "a", delta: 0.4 }], { width: 10, height: 10, step: 0.1 });
    expect(barPaths(only.bars).loss).toBe("");
  });

  it("gives a query that did not move a hairline, so it is still on the page", () => {
    expect(barPaths(bars, 0.8).flat).toContain("v0.8");
  });
});

describe("barIndexAt", () => {
  it("maps a position across the drawing to a bar", () => {
    expect(barIndexAt(0, 1000, 250)).toBe(0);
    expect(barIndexAt(500, 1000, 250)).toBe(125);
    expect(barIndexAt(999, 1000, 250)).toBe(249);
  });

  it("clamps rather than running off either end", () => {
    expect(barIndexAt(-40, 1000, 250)).toBe(0);
    expect(barIndexAt(4000, 1000, 250)).toBe(249);
  });

  it("has no bar to point at when there are none", () => {
    expect(barIndexAt(10, 1000, 0)).toBe(-1);
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

describe("candidateRows", () => {
  it("numbers the candidates from one, in the method's order", () => {
    const rows = candidateRows(detail.queries["Q-1"], "jev-score", "hybrid", ["D-2"], fixture.titles)!;
    expect(rows.map((r) => r.docId)).toEqual(["D-2", "D-3", "D-1"]);
    expect(rows.map((r) => r.rank)).toEqual([1, 2, 3]);
  });

  it("pairs each score with the document beside it in the order", () => {
    const rows = candidateRows(detail.queries["Q-1"], "jev-score", "hybrid", [], fixture.titles)!;
    expect(rows.map((r) => r.score)).toEqual([3.5, 1.25, 0.5]);
  });

  it("records where the floor had put each document, and how far it travelled", () => {
    const rows = candidateRows(detail.queries["Q-1"], "jev-score", "hybrid", [], fixture.titles)!;
    expect(rows.map((r) => r.floorRank)).toEqual([2, 3, 1]);
    expect(rows.map((r) => r.moved)).toEqual([1, 1, -2]);
  });

  it("marks the documents the judgements call relevant", () => {
    const rows = candidateRows(detail.queries["Q-1"], "hybrid", "hybrid", ["D-2"], fixture.titles)!;
    expect(rows.filter((r) => r.relevant).map((r) => r.docId)).toEqual(["D-2"]);
  });

  it("has no score to show for the floor, which never scored anything", () => {
    const rows = candidateRows(detail.queries["Q-1"], "hybrid", "hybrid", [], fixture.titles)!;
    expect(rows.every((r) => r.score === null)).toBe(true);
    expect(rows.every((r) => r.moved === 0)).toBe(true);
  });

  it("falls back to the document id when the run kept no title", () => {
    const rows = candidateRows(detail.queries["Q-1"], "hybrid", "hybrid", [], fixture.titles)!;
    expect(rows[2].title).toBe("D-3");
  });

  it("returns nothing when the method has no order for this query", () => {
    expect(candidateRows(detail.queries["Q-1"], "laya-score", "hybrid", [], fixture.titles)).toBeNull();
    expect(candidateRows(undefined, "jev-score", "hybrid", [], fixture.titles)).toBeNull();
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

describe("scoreNote", () => {
  it("says the floor never scored anything, because it is the retriever's order", () => {
    expect(scoreNote("hybrid", "hybrid", null)).toBe(
      "hybrid is the order the retriever returned, so it holds no score of its own.",
    );
  });

  it("says a scorer's missing number is a failed call, not an absent opinion", () => {
    expect(scoreNote("jev-score", "hybrid", null)).toBe(
      "jev-score returned nothing for this passage: the call failed, so it kept the slot hybrid gave it.",
    );
  });

  it("reads out the number when there is one", () => {
    expect(scoreNote("jev-score", "hybrid", 2.25)).toBe("jev-score scored this passage 2.250");
  });
});
