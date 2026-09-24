import { describe, expect, it } from "vitest";
import { columns, reasonLabel, type Example } from "./example";

/** A four-candidate stand-in with the shapes the exporter writes. */
const EX: Example = {
  query_id: "Q1",
  query: "does vitamin d prevent cancer",
  reason: "jev-gain",
  bm25: ["A", "B", "C", "D"],
  relevant: { C: 2, D: 1 },
  passages: {
    A: { title: "A", snippet: "a", chars: 1 },
    B: { title: "B", snippet: "b", chars: 1 },
    C: { title: "C", snippet: "c", chars: 1 },
    D: { title: "D", snippet: "d", chars: 1 },
  },
  methods: {
    m: { order: ["C", "A", "B", "D"], scores: { A: 2, B: 1, C: 3, D: null }, delta: 0.1, failed: ["D"] },
  },
  wire: { m: {} },
};

describe("columns", () => {
  const { bm25, method } = columns(EX, "m");

  it("keeps BM25 in its own order, one line per candidate", () => {
    expect(bm25.map((l) => l.docId)).toEqual(["A", "B", "C", "D"]);
    expect(bm25.map((l) => l.position)).toEqual([1, 2, 3, 4]);
  });

  it("puts the method's ordering beside it", () => {
    expect(method.map((l) => l.docId)).toEqual(["C", "A", "B", "D"]);
  });

  it("says how far each passage moved, up being positive", () => {
    const moves = Object.fromEntries(method.map((l) => [l.docId, l.move]));
    expect(moves).toEqual({ C: 2, A: -1, B: -1, D: 0 });
  });

  it("marks the passages the official judgements call relevant, with their grade", () => {
    expect(method.find((l) => l.docId === "C")).toMatchObject({ relevant: true, grade: 2 });
    expect(method.find((l) => l.docId === "A")).toMatchObject({ relevant: false, grade: 0 });
  });

  it("marks a call that never came back, and that passage did not move", () => {
    const d = method.find((l) => l.docId === "D")!;
    expect(d.failed).toBe(true);
    expect(d.score).toBeNull();
    expect(d.move).toBe(0);
  });

  it("carries the score the method actually returned", () => {
    expect(method.find((l) => l.docId === "C")!.score).toBe(3);
  });

  it("gives BM25 no move and no score: it is the reference, not a re-ranking", () => {
    expect(bm25.every((l) => l.move === 0 && l.score === null)).toBe(true);
  });

  it("is empty for a method the export does not carry", () => {
    expect(columns(EX, "missing").method).toEqual([]);
  });
});

describe("reasonLabel", () => {
  it("says why each example is in the subset, in the page's words", () => {
    expect(reasonLabel("jev-gain")).toMatch(/Jev/);
    expect(reasonLabel("laya-loss")).toMatch(/Laya/);
    expect(reasonLabel("unchanged")).toMatch(/moved|unchanged|nothing/i);
    expect(reasonLabel("jev-failure")).toMatch(/fail/i);
  });

  it("falls back to the raw reason rather than dropping it", () => {
    expect(reasonLabel("new-bucket")).toBe("new-bucket");
  });
});
