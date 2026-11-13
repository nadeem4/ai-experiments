/**
 * Which experiments the site publishes, and in what order.
 *
 * This is the only switch. An experiment marked `published: false` drops out of
 * the nav and the index, and its page renders the 404 instead of its results,
 * so a finished-but-unreviewed write-up cannot be reached by guessing its URL.
 * Flip the flag and rebuild to publish it.
 */
export interface ExperimentEntry {
  /** The route under app/, which is also the URL path. */
  slug: string;
  /** The short name the nav uses. */
  nav: string;
  published: boolean;
}

export const EXPERIMENTS: ExperimentEntry[] = [
  { slug: "rerank", nav: "Re-ranker", published: true },
];
