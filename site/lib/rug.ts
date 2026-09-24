/**
 * Distinct score values, counted into buckets across one method's own range.
 *
 * Jev's answers come back from the gateway rounded to two decimals, so they land
 * on a grid; the cross-encoder emits a raw logit and does not. Bucketing each
 * method over its own observed range puts the two on the same picture without
 * pretending a 0-to-4 score and a logit are the same quantity: what is being
 * compared is resolution, not level. The gaps between Jev's buckets are the
 * finding, so nothing is clamped into an edge bucket to tidy the ends up.
 */
export function rugBuckets(values: number[], min: number, max: number, bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  const span = max - min;
  for (const value of values) {
    if (value < min || value > max) continue;
    const at = span === 0 ? 0 : Math.min(bins - 1, Math.floor(((value - min) / span) * bins));
    counts[at] += 1;
  }
  return counts;
}
