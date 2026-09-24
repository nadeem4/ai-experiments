/**
 * The strip plot is the page's main claim, so its geometry is tested rather than
 * eyeballed. Two things have to hold: zero sits exactly on the zero line in every
 * row, and a column that holds many queries is taller than one that holds few,
 * because the pile of queries nothing moved is the point of the chart.
 */
import { describe, expect, it } from "vitest";
import { stripLayout, symmetricHalf, type StripRow } from "@/lib/strip";

const OPTIONS = { width: 1000, column: 4, band: 80, pitch: 4 };

function points(...values: number[]) {
  return values.map((value, i) => ({ id: `q${i}`, value, example: false }));
}

describe("symmetricHalf", () => {
  it("has a domain even when there is nothing to plot", () => {
    expect(symmetricHalf([], 0.1)).toBe(0.1);
  });

  it("covers the largest value in either direction", () => {
    expect(symmetricHalf([0.05, -0.3, 0.12], 0.1)).toBeCloseTo(0.3, 10);
  });

  it("rounds out to the next step so the axis lands on readable ticks", () => {
    expect(symmetricHalf([0.31], 0.1)).toBeCloseTo(0.4, 10);
    expect(symmetricHalf([-0.62], 0.25)).toBeCloseTo(0.75, 10);
  });
});

describe("stripLayout", () => {
  it("puts a query that did not move exactly on the zero line", () => {
    const rows: StripRow[] = [{ method: "jev-score", points: points(0) }];
    const [row] = stripLayout(rows, OPTIONS).rows;
    expect(row.points[0].x).toBe(500);
    expect(row.points[0].y).toBe(0);
  });

  it("puts a gain right of zero and a loss left of it", () => {
    const rows: StripRow[] = [{ method: "m", points: points(0.4, -0.4) }];
    const [row] = stripLayout(rows, { ...OPTIONS, column: 1 }).rows;
    expect(row.points[0].x).toBeGreaterThan(500);
    expect(row.points[1].x).toBe(1000 - row.points[0].x);
  });

  it("snaps to a column grid centred on zero, so equal values share a column", () => {
    const rows: StripRow[] = [{ method: "m", points: points(0.2, 0.2) }];
    const [row] = stripLayout(rows, OPTIONS).rows;
    expect(row.points[0].x).toBe(row.points[1].x);
  });

  it("stacks a shared column symmetrically about the row's centre line", () => {
    const rows: StripRow[] = [{ method: "m", points: points(0, 0, 0) }];
    const { pitch, rows: placed } = stripLayout(rows, OPTIONS);
    expect(placed[0].points.map((p) => p.y)).toEqual([-pitch, 0, pitch]);
  });

  it("tightens the pitch so the tallest column still fits the band", () => {
    const tall = points(...Array(40).fill(0));
    const { pitch, tallest } = stripLayout([{ method: "m", points: tall }], OPTIONS);
    expect(tallest).toBe(40);
    expect(pitch).toBeCloseTo(80 / 40, 10);
  });

  it("uses one pitch across every row, so a pile in one row reads against another", () => {
    const rows: StripRow[] = [
      { method: "a", points: points(...Array(40).fill(0)) },
      { method: "b", points: points(0, 0) },
    ];
    const { pitch, rows: placed } = stripLayout(rows, OPTIONS);
    expect(placed[1].points.map((p) => p.y)).toEqual([-pitch / 2, pitch / 2]);
  });

  it("puts the browsable queries at the outside of a stack, where they can be clicked", () => {
    const members = [
      { id: "a", value: 0, example: true },
      { id: "b", value: 0, example: false },
      { id: "c", value: 0, example: false },
      { id: "d", value: 0, example: true },
    ];
    const [row] = stripLayout([{ method: "m", points: members }], OPTIONS).rows;
    const at = Object.fromEntries(row.points.map((p) => [p.id, p.y]));
    expect(at.a).toBeLessThan(at.b);
    expect(at.d).toBeGreaterThan(at.c);
  });

  it("keeps every query it was given", () => {
    const rows: StripRow[] = [{ method: "m", points: points(0, 0.1, -0.1, 0, 0.9) }];
    expect(stripLayout(rows, OPTIONS).rows[0].points).toHaveLength(5);
  });
});
