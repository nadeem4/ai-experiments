import { Chip, Mono } from "@/components/chip";
import { byMethod, label, pct, spanBar, tieShare, type Results, type Scores } from "@/lib/results";

/**
 * Two of the findings above are easier to see than to state, and both are in the
 * shape of the scores rather than in the ranking metric. Plain bars: a chart
 * library would add a dependency to draw four rectangles.
 */
export function Distributions({ results }: { results: Results }) {
  const base = byMethod(results, "laya-score")!;
  const tuned = byMethod(results, "laya-typed-score")!;
  const jev = byMethod(results, "jev-score")!;
  const ce = byMethod(results, "cross-encoder")!;

  return (
    <div className="grid gap-8">
      <section aria-labelledby="span" className="grid gap-3">
        <h3 id="span" className="text-body font-extrabold">
          Why the fine-tune ranks below its own base
        </h3>
        <p className="max-w-[70ch] text-body leading-relaxed text-ink-soft">
          Both Laya checkpoints answer the same question on the same 0-to-4 scale, so their answers lie on one
          axis. The base checkpoint uses the whole of it; the <Mono>typed-decisions</Mono> fine-tune never
          touches the bottom third. A re-ranker that separates candidates less ranks them worse, and that is
          what the table shows.
        </p>
        <div className="grid gap-3 border border-line bg-surface p-4">
          <SpanRow name={label(base.method)} scores={base.scores!} />
          <SpanRow name={label(tuned.method)} scores={tuned.scores!} />
          <div className="numeric flex justify-between border-t border-line pt-2 text-micro text-ink-soft" aria-hidden>
            <span>0</span><span>1</span><span>2</span><span>3</span><span>4</span>
          </div>
          <p className="text-micro text-ink-soft">
            The shaded band is the bottom third of the scale. Standard deviation: {base.scores!.stdev} for the
            base checkpoint, {tuned.scores!.stdev} for the fine-tune.
          </p>
        </div>
      </section>

      <section aria-labelledby="resolution" className="grid gap-3">
        <h3 id="resolution" className="text-body font-extrabold">
          Why a quarter of Jev&apos;s passages never moved
        </h3>
        <p className="max-w-[70ch] text-body leading-relaxed text-ink-soft">
          The gateway rounds Jev&apos;s answers to two decimals, so a 0-to-4 score has at most 401 possible
          values. Jev used {jev.scores!.distinct} of them across {jev.scores!.of.toLocaleString()} successful
          calls; the cross-encoder emits a raw logit and returned {ce.scores!.distinct.toLocaleString()}{" "}
          distinct values across {ce.scores!.of.toLocaleString()}. Coarse steps mean ties, and a tie is not an
          opinion: tied passages keep the order BM25 gave them.
        </p>
        <div className="grid gap-4 border border-line bg-surface p-4">
          <TieRow name={label(jev.method)} scores={jev.scores!} />
          <TieRow name={label(ce.method)} scores={ce.scores!} />
          <p className="max-w-[70ch] text-micro leading-relaxed text-ink-soft">
            So Jev&apos;s gain is earned while roughly a quarter of its ranking is still BM25&apos;s. That cuts
            both ways, and this run does not separate them: it may mean Jev is decisive exactly where it
            matters, or that the gain comes from fewer decisions than the call count suggests. Its largest
            single tie was {jev.scores!.largest_group} of {results.top_k} candidates in one query.
          </p>
        </div>
      </section>
    </div>
  );
}

function SpanRow({ name, scores }: { name: string; scores: Scores }) {
  const bar = spanBar(scores.min, scores.max, 0, 4);
  return (
    <div className="grid gap-1">
      <p className="numeric flex flex-wrap items-baseline justify-between gap-2 text-micro">
        <span className="font-semibold">{name}</span>
        <span className="text-ink-soft">
          {scores.min.toFixed(3)} to {scores.max.toFixed(3)}
          <Chip>stdev {scores.stdev}</Chip>
        </span>
      </p>
      <div className="relative h-5 w-full bg-sunk" role="img"
        aria-label={`${name} answered between ${scores.min.toFixed(3)} and ${scores.max.toFixed(3)} on a 0 to 4 scale`}>
        <span aria-hidden className="absolute inset-y-0 left-0 w-1/3 border-r border-line-strong bg-line/60" />
        <span
          aria-hidden
          className="absolute inset-y-1 bg-accent"
          style={{ left: `${bar.left}%`, width: `${bar.width}%` }}
        />
      </div>
    </div>
  );
}

function TieRow({ name, scores }: { name: string; scores: Scores }) {
  const share = tieShare(scores);
  return (
    <div className="grid gap-1">
      <p className="numeric flex flex-wrap items-baseline justify-between gap-2 text-micro">
        <span className="font-semibold">{name}</span>
        <span className="text-ink-soft">
          {scores.tied.toLocaleString()} of {scores.of.toLocaleString()} passages tied
          <Chip>{pct(share)}</Chip>
        </span>
      </p>
      <div className="h-5 w-full bg-sunk" role="img"
        aria-label={`${pct(share)} of ${name} scored passages sit in a tie`}>
        <span className="block h-full bg-accent" style={{ width: `${Math.max(share * 100, 0.5)}%` }} />
      </div>
    </div>
  );
}
