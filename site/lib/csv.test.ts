/**
 * The CSV offered for download is built from the same per-query file the strip
 * plot is drawn from, so the download and the chart can never disagree. NFCorpus
 * queries are free text and some of them contain commas, so the quoting is the
 * part worth testing.
 */
import { describe, expect, it } from "vitest";
import { perQueryCsv } from "@/lib/csv";
import type { PerQuery } from "@/lib/per-query";

const DATA: PerQuery = {
  run: "test",
  queries: 2,
  methods: ["jev-score", "laya-score"],
  rows: [
    {
      id: "PLAIN-1",
      query: "milk",
      bm25: 0.25,
      deltas: { "jev-score": 0.1, "laya-score": -0.2 },
      relevant: 3,
      example: true,
    },
    {
      id: "PLAIN-2",
      query: 'Eggs, "cholesterol" and you',
      bm25: 0,
      deltas: { "jev-score": 0, "laya-score": 0 },
      relevant: 0,
      example: false,
    },
  ],
};

const lines = () => perQueryCsv(DATA).split("\n");

describe("perQueryCsv", () => {
  it("names every column, with one column per method", () => {
    expect(lines()[0]).toBe(
      "query_id,query,bm25_ndcg@10,jev-score,laya-score,judged_relevant_in_candidates,has_example",
    );
  });

  it("writes one row per query and no trailing blank line", () => {
    expect(lines()).toHaveLength(3);
  });

  it("writes the numbers as they are stored, not rounded again", () => {
    expect(lines()[1]).toBe("PLAIN-1,milk,0.25,0.1,-0.2,3,true");
  });

  it("quotes a query with a comma in it and doubles its quotes", () => {
    expect(lines()[2]).toBe('PLAIN-2,"Eggs, ""cholesterol"" and you",0,0,0,0,false');
  });

  it("writes a header even when there is nothing to write", () => {
    expect(perQueryCsv({ ...DATA, rows: [] }).split("\n")).toHaveLength(1);
  });
});
