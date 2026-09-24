/**
 * The rug lays each method's distinct score values on its own observed range, so
 * the two can be compared for resolution rather than for level. The bucketing is
 * what makes the gaps in Jev's grid visible at screen resolution, so the edges of
 * it are tested: the top value has to land inside the plot, not past its end.
 */
import { describe, expect, it } from "vitest";
import { rugBuckets } from "@/lib/rug";

describe("rugBuckets", () => {
  it("returns one count per bucket, all empty when there is nothing", () => {
    expect(rugBuckets([], 0, 1, 4)).toEqual([0, 0, 0, 0]);
  });

  it("puts the lowest value in the first bucket and the highest in the last", () => {
    expect(rugBuckets([0, 1], 0, 1, 4)).toEqual([1, 0, 0, 1]);
  });

  it("spreads values across the range", () => {
    expect(rugBuckets([0.1, 0.3, 0.6, 0.9], 0, 1, 4)).toEqual([1, 1, 1, 1]);
  });

  it("counts repeats in the same bucket", () => {
    expect(rugBuckets([0.1, 0.15, 0.2], 0, 1, 4)).toEqual([3, 0, 0, 0]);
  });

  it("ignores values outside the range rather than clamping them into an edge", () => {
    expect(rugBuckets([-1, 2, 0.5], 0, 1, 2)).toEqual([0, 1]);
  });

  it("leaves gaps where a method never answered, which is the whole point", () => {
    expect(rugBuckets([0, 0.5, 1], 0, 1, 10).filter((n) => n === 0)).toHaveLength(7);
  });

  it("survives a range of zero width", () => {
    expect(rugBuckets([2, 2], 2, 2, 3)).toEqual([2, 0, 0]);
  });
});
