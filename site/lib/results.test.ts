import { describe, expect, it } from "vitest";
import results from "@/data/results.json";
import { excludesZero, label, spanBar, tableRows, tieShare } from "./results";
import type { Results } from "./results";

const run = results as Results;

describe("excludesZero", () => {
  it("is false when the interval straddles zero, which is what makes a result noise", () => {
    expect(excludesZero([-0.0067, 0.0561])).toBe(false);
  });

  it("is true on both sides, because a real loss is a result too", () => {
    expect(excludesZero([0.0218, 0.0476])).toBe(true);
    expect(excludesZero([-0.0522, -0.0197])).toBe(true);
  });

  it("is false when there is no interval at all", () => {
    expect(excludesZero(null)).toBe(false);
    expect(excludesZero(undefined)).toBe(false);
  });
});

describe("tableRows", () => {
  const rows = tableRows(run);

  it("has one row per method in the run", () => {
    expect(rows).toHaveLength(run.methods.length);
  });

  it("orders them best nDCG@10 first, so the floor sits where it fell", () => {
    const ndcg = rows.map((r) => r.ndcg);
    expect(ndcg).toEqual([...ndcg].sort((a, b) => b - a));
  });

  it("marks the floor, and gives it no difference against itself", () => {
    const bm25 = rows.find((r) => r.method === "bm25")!;
    expect(bm25.isFloor).toBe(true);
    expect(bm25.delta).toBeNull();
  });

  it("carries the paired difference and whether its interval clears zero", () => {
    const jev = rows.find((r) => r.method === "jev-score")!;
    expect(jev.delta).toBeCloseTo(0.0347, 4);
    expect(jev.excludesZero).toBe(true);
    expect(jev.better).toBe(117);
    expect(jev.worse).toBe(52);
  });

  it("keeps Laya negative, because that is the finding", () => {
    const laya = rows.find((r) => r.method === "laya-score")!;
    expect(laya.delta).toBeLessThan(0);
    expect(laya.excludesZero).toBe(true);
  });

  it("every method in this run separated from the floor", () => {
    expect(rows.filter((r) => !r.isFloor).every((r) => r.excludesZero)).toBe(true);
  });
});

describe("label", () => {
  it("names the methods the way the run does, without inventing new ones", () => {
    expect(label("jev-score")).toBe("Jev score");
    expect(label("laya-typed-score")).toBe("Laya typed-decisions score");
    expect(label("bm25")).toBe("BM25 (the floor)");
    expect(label("something-else")).toBe("something-else");
  });
});

describe("tieShare", () => {
  it("is the share of scored passages sitting in a tie", () => {
    expect(tieShare({ tied: 1748, of: 6147 } as never)).toBeCloseTo(0.284, 3);
  });

  it("is zero when nothing was scored", () => {
    expect(tieShare(undefined)).toBe(0);
  });
});

describe("spanBar", () => {
  it("lays a range on the axis as a percentage of it", () => {
    expect(spanBar(1, 3, 0, 4)).toEqual({ left: 25, width: 50 });
  });

  it("starts at the left edge when the range does", () => {
    expect(spanBar(0, 2, 0, 4)).toEqual({ left: 0, width: 50 });
  });

  it("clamps a range that runs off the axis rather than overflowing the row", () => {
    expect(spanBar(-1, 9, 0, 4)).toEqual({ left: 0, width: 100 });
  });

  it("keeps a hairline for a range with no width, so it is still visible", () => {
    expect(spanBar(2, 2, 0, 4).width).toBeGreaterThan(0);
  });

  it("shows the fine-tune never reaching the bottom third of the scale", () => {
    const base = run.methods.find((m) => m.method === "laya-score")!.scores!;
    const tuned = run.methods.find((m) => m.method === "laya-typed-score")!.scores!;
    expect(spanBar(base.min, base.max, 0, 4).left).toBeLessThan(100 / 3);
    expect(spanBar(tuned.min, tuned.max, 0, 4).left).toBeGreaterThan(100 / 3);
  });
});
