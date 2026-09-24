/**
 * The slope chart draws one line per passage between the position BM25 gave it
 * and the position the re-ranker gave it. Everything that can go wrong is in the
 * position mapping, so that is what is tested: a passage that did not move has to
 * draw a flat line, and the judgements have to follow the passage, not the slot.
 */
import { describe, expect, it } from "vitest";
import { slopeLines } from "@/lib/slope";

const BM25 = ["a", "b", "c"];

describe("slopeLines", () => {
  it("draws nothing when the re-ranker returned nothing", () => {
    expect(slopeLines(BM25, [], {})).toEqual([]);
  });

  it("numbers positions from one, in BM25's order", () => {
    const lines = slopeLines(BM25, BM25, {});
    expect(lines.map((l) => l.docId)).toEqual(["a", "b", "c"]);
    expect(lines.map((l) => l.from)).toEqual([1, 2, 3]);
  });

  it("leaves an unchanged ranking flat", () => {
    expect(slopeLines(BM25, BM25, {}).every((l) => l.from === l.to && l.moved === 0)).toBe(true);
  });

  it("counts places moved up as positive", () => {
    const lines = slopeLines(BM25, ["c", "a", "b"], {});
    const moved = Object.fromEntries(lines.map((l) => [l.docId, l.moved]));
    expect(moved).toEqual({ a: -1, b: -1, c: 2 });
  });

  it("reads the new position from the re-ranked order", () => {
    const lines = slopeLines(BM25, ["c", "a", "b"], {});
    expect(Object.fromEntries(lines.map((l) => [l.docId, l.to]))).toEqual({ a: 2, b: 3, c: 1 });
  });

  it("carries the judgement with the passage", () => {
    const lines = slopeLines(BM25, ["c", "a", "b"], { c: 2, b: 0 });
    const judged = Object.fromEntries(lines.map((l) => [l.docId, [l.relevant, l.grade]]));
    expect(judged).toEqual({ a: [false, 0], b: [false, 0], c: [true, 2] });
  });

  it("keeps a passage the re-ranker never returned at its BM25 position", () => {
    const lines = slopeLines(BM25, ["a", "b"], {});
    expect(lines.find((l) => l.docId === "c")).toMatchObject({ from: 3, to: 3, moved: 0 });
  });
});
