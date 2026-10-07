/**
 * "Watch one question": what one question's rankings say, read the way the
 * section shows it. Answers are named by where the search put them, every
 * measure is read off a ranking's own order, and the one-line verdict never
 * claims a win the scores do not show.
 */
import { describe, expect, it } from "vitest";
import { answerOrder, cellIdsFit, questionMeasures, verdictLine } from "@/lib/rerank-watch";

const name = (m: string) => ({ "jev-score": "Jev", "cross-encoder": "Cross-encoder", "laya-score": "Laya" })[m] ?? m;
const text = (parts: { text: string }[]) => parts.map((p) => p.text).join("");

// Five candidates, two of them answers; the search put them 2nd and 4th.
const search = ["d1", "d2", "d3", "d4", "d5"];
const orders = {
  hybrid: search,
  "jev-score": ["d4", "d2", "d1", "d3", "d5"],
  "laya-score": ["d1", "d3", "d5", "d2", "d4"],
};
const relevant = ["d4", "d2"];

describe("answerOrder", () => {
  it("numbers the answers by where the search put them, so A1 is the search's highest", () => {
    expect(answerOrder(relevant, search)).toEqual(["d2", "d4"]);
  });

  it("is empty when nothing among the candidates is an answer", () => {
    expect(answerOrder([], search)).toEqual([]);
  });
});

describe("questionMeasures", () => {
  it("reads the first answer's position and its reciprocal from the order", () => {
    const m = questionMeasures(orders["jev-score"], relevant);
    expect(m.first).toBe(1);
    expect(m.rr).toBe(1);
    expect(m.inTop).toBe(2);
  });

  it("gives no reciprocal rank when the first answer falls outside the top ten", () => {
    const deep = Array.from({ length: 20 }, (_, i) => `x${i}`);
    deep[14] = "a";
    expect(questionMeasures(deep, ["a"])).toEqual({ first: 15, rr: 0, inTop: 0 });
  });

  it("has no first answer when there is none to find", () => {
    expect(questionMeasures(search, [])).toEqual({ first: null, rr: 0, inTop: 0 });
  });
});

describe("cellIdsFit", () => {
  it("numbers strip cells only while every answer id is a single digit", () => {
    expect(cellIdsFit(9)).toBe(true);
    expect(cellIdsFit(10)).toBe(false);
  });
});

describe("verdictLine", () => {
  const ndcg = { hybrid: 0.5, "jev-score": 1, "laya-score": 0.2 };

  it("names the best re-ranker and where it moved the first answer", () => {
    const line = text(verdictLine(orders, relevant, ndcg, "hybrid", ["jev-score", "laya-score"], name));
    expect(line).toContain("Jev did best here");
    expect(line).toContain("from 02 to 01");
    expect(line).toContain("0.500 became 1.000");
  });

  it("names the worst re-ranker only when it fell below the search", () => {
    const line = text(verdictLine(orders, relevant, ndcg, "hybrid", ["jev-score", "laya-score"], name));
    expect(line).toContain("Laya did worst");
  });

  it("sets the re-rankers' names apart, so the page can make them bold", () => {
    const parts = verdictLine(orders, relevant, ndcg, "hybrid", ["jev-score", "laya-score"], name);
    expect(parts.filter((p) => p.strong).map((p) => p.text)).toEqual(["Jev", "Laya"]);
  });

  it("says so plainly when no re-ranker beat the search", () => {
    const flat = { hybrid: 0.5, "jev-score": 0.5, "laya-score": 0.4 };
    expect(text(verdictLine(orders, relevant, flat, "hybrid", ["jev-score", "laya-score"], name))).toContain(
      "No re-ranker beat the search",
    );
  });

  it("says there is nothing to find when the candidates hold no answer", () => {
    expect(text(verdictLine(orders, [], ndcg, "hybrid", ["jev-score"], name))).toContain("Nothing to find");
  });

  it("joins tied winners into one name", () => {
    const tied = { hybrid: 0.5, "jev-score": 1, "laya-score": 1 };
    const both = { ...orders, "laya-score": orders["jev-score"] };
    const parts = verdictLine(both, relevant, tied, "hybrid", ["jev-score", "laya-score"], name);
    expect(parts.find((p) => p.strong)?.text).toBe("Jev and Laya");
    expect(text(parts)).not.toContain("did worst");
  });
});
