/**
 * Reading one question's rankings for "Watch one question being re-ranked".
 *
 * Everything here is worked out from the orders the run recorded and the
 * question's own nDCG@10, so the section can say what happened to a question
 * without a number the results file does not support.
 */

const TOP = 10;
// An answer id fits inside a strip cell only while it is a single digit.
const MAX_CELL_IDS = 9;

export interface QuestionMeasures {
  /** Position of the highest-placed answer, 1-based; null when there is none to find. */
  first: number | null;
  /** 1 / first while the first answer is in the top ten, else 0. */
  rr: number;
  /** Answers placed in the top ten. */
  inTop: number;
}

/** The answers, in the order the search put them: index 0 is A1. */
export function answerOrder(relevant: string[], searchOrder: string[]): string[] {
  return [...relevant].sort((a, b) => searchOrder.indexOf(a) - searchOrder.indexOf(b));
}

export function questionMeasures(order: string[], relevant: string[]): QuestionMeasures {
  const ranks = relevant.map((d) => order.indexOf(d) + 1).filter((r) => r > 0);
  const first = ranks.length ? Math.min(...ranks) : null;
  return { first, rr: first !== null && first <= TOP ? 1 / first : 0, inTop: ranks.filter((r) => r <= TOP).length };
}

export const cellIdsFit = (answers: number) => answers <= MAX_CELL_IDS;

export interface Segment {
  text: string;
  strong?: boolean;
}

const pad = (n: number) => String(n).padStart(2, "0");
const joinNames = (names: string[]) =>
  names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;

/**
 * One sentence on what happened to this question: who did best and where the
 * first answer went, and who did worst when anyone fell below the search.
 * Returned as segments so the re-rankers' names can be set in bold.
 */
export function verdictLine(
  orders: Record<string, string[]>,
  relevant: string[],
  ndcg: Record<string, number>,
  floor: string,
  rerankers: string[],
  name: (method: string) => string,
): Segment[] {
  if (!relevant.length) {
    return [{ text: "Nothing to find here: none of the 20 candidates answers the question, so every ranking scores 0 whatever order it picks." }];
  }
  const base = ndcg[floor];
  const firstAt = (m: string) => questionMeasures(orders[m], relevant).first ?? 0;
  const best = Math.max(...rerankers.map((m) => ndcg[m]));
  const worst = Math.min(...rerankers.map((m) => ndcg[m]));
  const out: Segment[] = [];

  if (best > base + 1e-9) {
    const winners = rerankers.filter((m) => ndcg[m] === best);
    const from = firstAt(floor);
    const to = firstAt(winners[0]);
    const move =
      to < from ? `lifting the first answer from ${pad(from)} to ${pad(to)}` : to === from ? `keeping the first answer at ${pad(from)}` : `even with the first answer at ${pad(to)}`;
    out.push({ text: joinNames(winners.map(name)), strong: true }, { text: ` did best here, ${move}: ${base.toFixed(3)} became ${best.toFixed(3)}.` });
  } else {
    out.push({ text: `No re-ranker beat the search on this question (${base.toFixed(3)}).` });
  }

  if (worst < base - 1e-9) {
    const losers = rerankers.filter((m) => ndcg[m] === worst);
    out.push({ text: " " }, { text: joinNames(losers.map(name)), strong: true }, {
      text: ` did worst, with the first answer at ${pad(firstAt(losers[0]))} and a score of ${worst.toFixed(3)}.`,
    });
  }
  return out;
}
