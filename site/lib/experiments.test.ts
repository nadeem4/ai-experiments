import { describe, expect, it } from "vitest";
import bankingData from "@/data/banking77.json";
import typedData from "@/data/typed-decisions.json";
import { describeValidity, pct, points, type Banking, type TypedDecisions } from "./experiments";

const banking = bankingData as Banking;
const typed = typedData as unknown as TypedDecisions;

describe("points", () => {
  it("keeps the sign, so a result that came out negative reads as negative", () => {
    expect(points(4.81)).toBe("+4.81");
    expect(points(-9.61)).toBe("−9.61");
  });
});

describe("describeValidity", () => {
  it("puts valid first and names every other outcome by its kind", () => {
    expect(describeValidity({ valid: 276, api_error: 24 })).toBe("276 valid, 24 api_error");
    expect(describeValidity({ api_error: 300 })).toBe("300 api_error");
  });
});

describe("pct", () => {
  it("reads a share as a percentage", () => {
    expect(pct(0.1118, 1)).toBe("11.2%");
  });
});

/**
 * The page is prose over generated data, so every sentence that asserts
 * something is pinned here. If a re-run moves a number far enough to make a
 * sentence false, this suite fails rather than the page quietly lying.
 */
describe("what the banking77 page says", () => {
  const arm = (name: string) => banking.arms.find((a) => a.arm === name)!;

  it("the default arm reproduces the published number, so the instrument works", () => {
    const [low, high] = arm("A-default").ci95;
    expect(banking.published.laya).toBeGreaterThan(low);
    expect(banking.published.laya).toBeLessThan(high);
  });

  it("the budget buys a real gain that is a small share of the gap", () => {
    expect(banking.headline.budget_gain_points).toBeGreaterThan(0);
    expect(banking.headline.share_of_deficit).toBeLessThan(0.2);
    expect(banking.headline.remaining_points).toBeGreaterThan(30);
  });

  it("the documented workaround lost to both the default and the raised budget", () => {
    expect(banking.headline.workaround_vs_default_points).toBeLessThan(0);
    expect(banking.headline.workaround_vs_raised_points).toBeLessThan(0);
  });

  it("the raised-budget arms agree to every decimal: a cliff, not a dial", () => {
    const raised = banking.headline.identical_budget_arms.map((name) => arm(name).accuracy);
    expect(raised.length).toBeGreaterThan(1);
    expect(new Set(raised).size).toBe(1);
  });

  it("accuracy falls with the option count while nothing below 77 truncates", () => {
    const sweep = [...banking.option_sweep].sort((a, b) => a.n_options - b.n_options);
    const accuracy = sweep.map((row) => row.accuracy);
    expect(accuracy).toEqual([...accuracy].sort((a, b) => b - a));
    expect(sweep.slice(0, -1).every((row) => row.identical_pairs === 0)).toBe(true);
  });

  it("the English checkpoint's interval does not contain the published number", () => {
    const [low, high] = arm("E-english").ci95;
    expect(banking.published.laya < low || banking.published.laya > high).toBe(true);
  });

  it("every latency in the run is the same device, so none is a cross-device claim", () => {
    expect(banking.device).toBe("cpu");
  });
});

describe("what the typed-decisions page says", () => {
  const hosted = (task: string) =>
    typed.models.filter((m) => m.task === task && !m.local && m.p50_ms !== null);
  const scored = (task: string) => typed.models.filter((m) => m.task === task && m.accuracy !== null);

  it("Jev is the fastest hosted model on both tasks", () => {
    for (const task of ["ag_news", "clinc150"]) {
      const fastest = hosted(task).reduce((a, b) => (a.p50_ms! < b.p50_ms! ? a : b));
      expect(fastest.model).toBe("jev");
    }
  });

  it("and it wins on accuracy on neither", () => {
    for (const task of ["ag_news", "clinc150"]) {
      const best = scored(task).reduce((a, b) => (a.accuracy! > b.accuracy! ? a : b));
      expect(best.model).not.toBe("jev");
    }
  });

  it("Jev's median barely moves from 4 options to 151, and it is not alone in that", () => {
    const rise = (model: string) => typed.latency_rise.find((r) => r.model === model)!.rise;
    expect(Math.abs(rise("jev"))).toBeLessThan(0.1);
    // glm and phi move as little. Flat is not unique to the decision model; flat
    // and fastest is, and the page says it that way round.
    expect(Math.abs(rise("glm"))).toBeLessThan(0.1);
    expect(rise("gemma")).toBeGreaterThan(0.5);
  });

  it("Laya's 151-option row is a refusal: no accuracy, no latency, nothing valid", () => {
    const laya = typed.models.find((m) => m.model === "laya" && m.task === "clinc150")!;
    expect(laya.valid_rate).toBe(0);
    expect(laya.accuracy).toBeNull();
    expect(laya.p50_ms).toBeNull();
  });

  it("one of the eighteen bias rows moves accuracy by more than a tenth, and no other by more than five points", () => {
    expect(typed.position_bias).toHaveLength(18);
    const moved = typed.position_bias.filter((b) => (b.spread ?? 0) > 0.1);
    expect(moved.map((b) => `${b.model}/${b.task}`)).toEqual(["phi/clinc150"]);
    const rest = typed.position_bias.filter((b) => !(b.model === "phi" && b.task === "clinc150"));
    expect(Math.max(...rest.map((b) => b.spread ?? 0))).toBeLessThan(0.06);
  });

  it("the bias arm is forty examples per placement, which is what limits it", () => {
    expect(new Set(typed.position_bias.map((b) => b.n_examples))).toEqual(new Set([40]));
  });

  it("at 151 options the option list is most of what an LLM is billed for", () => {
    const wide = typed.option_share.find((s) => s.n_options === 151)!;
    expect(wide.min_share).toBeGreaterThan(0.8);
  });

  it("temperature scaling made calibration worse on test in every fit", () => {
    expect(typed.calibration.every((c) => c.ece_scaled > c.ece_raw)).toBe(true);
  });

  it("two models return a distribution and the seven LLMs return a label", () => {
    expect(typed.probability.returns).toEqual(["jev", "laya"]);
    expect(typed.probability.label_only).toHaveLength(7);
  });
});
