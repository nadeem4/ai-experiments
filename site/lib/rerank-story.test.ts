/**
 * The story the page tells is one question followed from search to grade, then
 * the whole run graded eight ways. What is tested is that every sentence it
 * builds is read out of the run: the question is the one the Watch section opens
 * on, the places are where the orders really put the answer, and a verdict is
 * only ever what that measure's interval allows.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import rerankData from "@/data/rerank.json";
import type { Detail, Rerank } from "@/lib/rerank";
import {
  MEASURES,
  measureBoard,
  ordinal,
  pairOn,
  placeWorth,
  storyMethods,
  storyQuestion,
  storyQuestionId,
} from "@/lib/rerank-story";

const run = rerankData as Rerank;
const detail = JSON.parse(readFileSync(join(process.cwd(), "public/rerank-detail.json"), "utf8")) as Detail;

describe("the question the story follows", () => {
  it("is followed by the best re-ranker and the worst", () => {
    expect(storyMethods(run)).toEqual({ helped: "jev-score", hurt: "laya-typed-score" });
  });

  it("is the question where those two disagree most, which is shelf life in this run", () => {
    expect(storyQuestionId(run)).toBe("PLAIN-2081");
  });

  it("follows the relevant abstract the search ranked highest, from where the search put it", () => {
    const story = storyQuestion(run, detail);
    expect(story.text).toBe("shelf life");
    expect(story.results).toHaveLength(run.summary.top_k);
    expect(story.results[story.searchAt - 1].id).toBe(story.answer);
    expect(story.searchAt).toBe(7);
  });

  it("shows each result by its title, as a person scanning results reads it, not by its passage text", () => {
    const story = storyQuestion(run, detail);
    expect(story.results[story.searchAt - 1].title).toBe(
      "Oxidative stability and shelf-life evaluation of selected culinary oils.",
    );
  });

  it("knows where each of the two re-rankers moved it, and what that order scored", () => {
    const story = storyQuestion(run, detail);
    expect(story.helped.at).toBe(1);
    expect(story.helped.ndcg).toBeCloseTo(1, 4);
    expect(story.helped.order[0]).toBe(story.answer);
    expect(story.helped.scores).toHaveLength(run.summary.top_k);
    expect(story.searchNdcg).toBeCloseTo(1 / 3, 3);
  });

  it("names the re-ranker that buried the answer by the lowest place any of them gave it", () => {
    const story = storyQuestion(run, detail);
    expect(story.buried.at).toBe(20);
    expect(story.buried.method).toBe("laya-score");
    expect(story.buried.ndcg).toBe(0);
  });
});

describe("words for places", () => {
  it("spells the first twenty", () => {
    expect([1, 2, 7, 12, 20].map(ordinal)).toEqual(["first", "second", "seventh", "twelfth", "twentieth"]);
  });

  it("falls back to digits past twenty", () => {
    expect(ordinal(21)).toBe("21st");
  });
});

describe("what a place is worth", () => {
  it("is nDCG's own discount in the top ten and nothing below it", () => {
    expect(placeWorth(1)).toBe(1);
    expect(placeWorth(7)).toBeCloseTo(1 / 3, 6);
    expect(placeWorth(11)).toBe(0);
  });
});

describe("grading the run eight ways", () => {
  it("offers every measure the run reports, the headline first", () => {
    expect(MEASURES[0].key).toBe("ndcg@10");
    expect(MEASURES.map((m) => m.key).sort()).toEqual(Object.keys(run.summary.by_metric!).sort());
  });

  it("keeps the re-rankers in the headline order whatever the measure, so the rows never jump", () => {
    const order = (key: string) => measureBoard(run, key)!.rows.map((r) => r.method);
    expect(order("success@1")).toEqual(order("ndcg@10"));
  });

  it("reads each value and the floor's from the run", () => {
    const board = measureBoard(run, "success@1")!;
    expect(board.floor).toBe(0.4458);
    expect(board.rows.find((r) => r.method === "jev-score")!.value).toBe(0.5046);
  });

  it("calls a difference only what its interval allows", () => {
    const mrr = measureBoard(run, "mrr@10")!;
    expect(mrr.rows.find((r) => r.method === "jev-score")!.verdict).toBe("better");
    expect(mrr.rows.find((r) => r.method === "cross-encoder")!.verdict).toBe("no clear change");
    expect(mrr.rows.find((r) => r.method === "laya-score")!.verdict).toBe("worse");
  });

  it("has nothing to draw for a measure the run did not report", () => {
    expect(measureBoard(run, "ndcg@100")).toBeUndefined();
  });
});

describe("two re-rankers against each other on one measure", () => {
  it("reads the pair as the run wrote it", () => {
    const pair = pairOn(run, "ndcg@10", "jev-score", "cross-encoder")!;
    expect(pair.mean).toBe(0.0319);
    expect(pair.better).toBe(124);
  });

  it("turns the pair when it is asked the other way round", () => {
    const pair = pairOn(run, "ndcg@10", "cross-encoder", "jev-score")!;
    expect(pair.mean).toBe(-0.0319);
    expect(pair.ci95).toEqual([-0.045, -0.0188]);
    expect(pair.better).toBe(67);
  });

  it("is missing rather than invented for a pair or a measure the run never compared", () => {
    expect(pairOn(run, "ndcg@10", "jev-score", "bm25")).toBeUndefined();
    expect(pairOn(run, "ndcg@100", "jev-score", "cross-encoder")).toBeUndefined();
  });

  it("finds that Jev and the cross-encoder tie on the first result", () => {
    const pair = pairOn(run, "success@1", "jev-score", "cross-encoder")!;
    expect(pair.ci95![0]).toBeLessThan(0);
    expect(pair.ci95![1]).toBeGreaterThan(0);
  });
});
