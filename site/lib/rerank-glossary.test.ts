/**
 * The glossary is read by people who do not know the field, so its rules are
 * about the reader: short, plain, sourced, and never a number the run did not
 * produce.
 */
import { describe, expect, it } from "vitest";
import rerankData from "@/data/rerank.json";
import { GLOSSARY, type TermId } from "@/lib/rerank-glossary";
import { floorRow, rerankerRows, type Rerank } from "@/lib/rerank";

const run = rerankData as Rerank;
const entries = Object.entries(GLOSSARY);
const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

describe("GLOSSARY", () => {
  it("explains every term in at most 60 words, so a card never needs scrolling", () => {
    for (const [, entry] of entries) expect(words(entry.says)).toBeLessThanOrEqual(60);
  });

  it("uses no em or en dash anywhere a reader sees", () => {
    for (const [, entry] of entries) {
      const visible = [entry.term, entry.kind, entry.says, entry.here ?? "", entry.link?.label ?? ""].join(" ");
      expect(visible).not.toMatch(/[–—]/);
    }
  });

  it("links only to https pages", () => {
    for (const [, entry] of entries) if (entry.link) expect(entry.link.href).toMatch(/^https:\/\//);
  });

  it("reads Jev's figures from the run rather than from prose", () => {
    const jev = rerankerRows(run).find((r) => r.method === "jev-score")!;
    expect(GLOSSARY.jev.here).toContain(jev.ndcg.toFixed(3));
  });

  it("reads the floor from the run", () => {
    expect(GLOSSARY.floor.here).toContain(floorRow(run)!["ndcg@10"].toFixed(3));
  });

  it("covers every term the page wraps", () => {
    const used: TermId[] = ["jev", "laya", "cross-encoder", "encoder", "bm25", "rrf", "ndcg", "floor", "nfcorpus", "interval", "p50", "rerank"];
    for (const id of used) expect(GLOSSARY[id]).toBeDefined();
  });
});
