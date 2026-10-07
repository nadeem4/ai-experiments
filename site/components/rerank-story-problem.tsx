import type { StoryQuestion } from "@/lib/rerank-story";
import { ordinal } from "@/lib/rerank-story";

const SHOWN = 10;
const pad = (n: number) => String(n).padStart(2, "0");

/**
 * The first screen: one real search, and the right answer sitting lower than it
 * should. It is the page's h1 because it is the page's claim: search finds, but
 * does not order.
 *
 * The arrival is CSS only (globals.css, "The story"): the query types itself,
 * the results land, the answer lights, the headline follows. Nothing here is
 * stateful, so it renders complete on the server.
 */
export function RerankStoryProblem({
  story,
  documents,
  questions,
  rerankers,
}: {
  story: StoryQuestion;
  documents: number;
  questions: number;
  rerankers: number;
}) {
  // When the typing ends, everything after it starts.
  const typed = 300 + story.text.length * 70 + 250;
  const answerAt = story.searchAt;

  return (
    <section
      aria-labelledby="problem"
      className="grid min-h-[calc(100svh-4rem)] content-center gap-7 py-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:items-center lg:gap-14"
      style={{ "--t0": `${typed}ms` } as React.CSSProperties}
    >
      <div className="grid content-start gap-7">
        <p className="text-micro text-ink-soft">An experiment in re-ranking search results</p>
        <div className="grid gap-1.5">
          <p className="text-[0.9375rem] text-ink-soft">
            Someone searches <span className="numeric">{documents.toLocaleString("en-US")}</span> medical abstracts
            for
          </p>
          <p className="story-typed flex min-h-14 items-center border-[1.5px] border-ink bg-surface px-4 text-2xl">
            <span className="sr-only">{story.text}</span>
            <span aria-hidden>
              {[...story.text].map((ch, i) => (
                <span key={i} style={{ "--i": i } as React.CSSProperties}>
                  {ch}
                </span>
              ))}
            </span>
            <span className="story-caret" aria-hidden />
          </p>
        </div>
        <div className="story-after">
          <h1 id="problem" className="text-balance text-[clamp(2rem,1.3rem+3vw,3.4rem)] font-semibold leading-[1.08] tracking-tight">
            The search found the right abstract. It put it <span className="text-mark">{ordinal(answerAt)}</span>.
          </h1>
          <p className="mt-5 max-w-[40ch] text-lead leading-relaxed text-ink-soft">
            Search is good at <b className="font-semibold text-ink">finding</b> and bad at{" "}
            <b className="font-semibold text-ink">ordering</b>. A second, slower reader could put the best ones
            first. We tested {rerankers === 4 ? "four" : rerankers} of them on {questions} questions.
          </p>
        </div>
      </div>

      <div>
        <ol className="grid gap-[3px]" aria-label={`The search's first ${SHOWN} results`}>
          {story.results.slice(0, SHOWN).map((result, i) => {
            const isAnswer = result.id === story.answer;
            return (
              <li
                key={result.id}
                style={{ "--i": i } as React.CSSProperties}
                className={`story-result grid min-h-[34px] items-center gap-2 border px-2.5 py-1 text-[0.9375rem] leading-snug ${
                  isAnswer
                    ? "answer grid-cols-[2rem_minmax(0,1fr)_auto] border-mark bg-mark-wash"
                    : "grid-cols-[2rem_minmax(0,1fr)] border-line bg-surface"
                }`}
              >
                <span className={`numeric text-xs ${isAnswer ? "font-medium text-mark" : "text-ink-soft"}`}>
                  {pad(i + 1)}
                </span>
                <span className={`truncate ${isAnswer ? "font-semibold" : ""}`}>{result.title}</span>
                {isAnswer && (
                  <span className="numeric whitespace-nowrap bg-mark px-1.5 text-[11px] font-medium text-page">
                    the right answer
                  </span>
                )}
              </li>
            );
          })}
        </ol>
        <p className="story-result px-2.5 pt-1.5 text-micro text-ink-soft" style={{ "--i": SHOWN } as React.CSSProperties}>
          and {story.results.length - SHOWN} more
        </p>
      </div>
    </section>
  );
}
