/**
 * Where each of the 323 queries sits in the strip plot.
 *
 * The chart exists because a mean and an interval hide the shape of the result.
 * Two things about that shape have to survive the drawing. Zero is a real
 * category here -- a query where the re-ranker changed nothing, or where BM25
 * retrieved nothing judged relevant to change -- so the domain is centred on
 * zero and the column grid is laid out from the centre, which puts those queries
 * exactly on the line rather than a rounding error away from it. And there are
 * far more of them than any single dot can show, so queries that share a column
 * stack, and the stack is what makes the pile visible.
 *
 * The pitch is worked out once across every row rather than per row, so a tall
 * column in one method means the same number of queries as a tall column in
 * another.
 */

export interface StripPoint {
  id: string;
  value: number;
  /** Whether this query's full wire was exported, so a reader can open it. */
  example: boolean;
}

export interface StripRow {
  method: string;
  points: StripPoint[];
}

export interface PlacedPoint extends StripPoint {
  x: number;
  /** Offset from the row's centre line; positive is down. */
  y: number;
}

export interface StripLayout {
  half: number;
  pitch: number;
  tallest: number;
  rows: { method: string; points: PlacedPoint[] }[];
}

export interface StripOptions {
  width: number;
  /** Width of one column of the grid dots snap to. */
  column: number;
  /** Height available to one row's stack. */
  band: number;
  /** Distance between two dots in a column, before the band caps it. */
  pitch: number;
  step?: number;
}

/** Half-width of a domain centred on zero that covers every value, rounded out
 * to the next `step` so the axis lands on ticks a reader can name. */
export function symmetricHalf(values: number[], step = 0.1): number {
  const extent = Math.max(0, ...values.map(Math.abs));
  return Math.max(step, Math.ceil(extent / step - 1e-9) * step);
}

/** Slots, top to bottom, for the members of one column. The browsable queries
 * take the outermost slots, alternating top and bottom: in a column of 150 the
 * dots are far closer together than a finger, and the ones worth clicking are
 * the only ones that have anywhere to go. */
function slots(members: PlacedPoint[]): PlacedPoint[] {
  const ordered: (PlacedPoint | undefined)[] = new Array(members.length);
  const examples = members.filter((m) => m.example);
  const rest = members.filter((m) => !m.example);
  let top = 0;
  let bottom = members.length - 1;
  examples.forEach((member, i) => {
    ordered[i % 2 === 0 ? top++ : bottom--] = member;
  });
  for (let i = 0; i < ordered.length; i++) if (!ordered[i]) ordered[i] = rest.shift();
  return ordered as PlacedPoint[];
}

/** Every query placed, with one pitch shared by every row. */
export function stripLayout(rows: StripRow[], options: StripOptions): StripLayout {
  const { width, column, band, step } = options;
  const half = symmetricHalf(rows.flatMap((r) => r.points.map((p) => p.value)), step);
  const centre = width / 2;

  // Columns are counted out from the centre, so a value of exactly zero lands on
  // the zero line rather than in whichever column the left edge happened to make.
  const columns = rows.map((row) => {
    const grouped = new Map<number, PlacedPoint[]>();
    for (const point of row.points) {
      const index = Math.round(((point.value / half) * centre) / column);
      const at = grouped.get(index);
      const placed = { ...point, x: centre + index * column, y: 0 };
      if (at) at.push(placed);
      else grouped.set(index, [placed]);
    }
    return grouped;
  });

  const tallest = Math.max(
    1,
    ...columns.flatMap((grouped) => [...grouped.values()].map((m) => m.length)),
  );
  const pitch = Math.min(options.pitch, band / tallest);

  return {
    half,
    pitch,
    tallest,
    rows: rows.map((row, i) => ({
      method: row.method,
      points: [...columns[i].values()].flatMap((members) =>
        slots(members).map((member, slot) => ({
          ...member,
          y: (slot - (members.length - 1) / 2) * pitch,
        })),
      ),
    })),
  };
}
