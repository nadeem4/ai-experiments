import { Term } from "@/components/term";
import { placeWorth, type StoryQuestion } from "@/lib/rerank-story";

const pad = (n: number) => String(n).padStart(2, "0");
const TOP = 10;

// Who each re-ranker is, in a sentence, from the protocol (rerank/README.md).
const WHO: Record<string, { kind: string; text: string; yardstick?: boolean }> = {
  "jev-score": {
    kind: "hosted, by TypeSafe AI",
    text: "A closed model reached over the internet. Asked of each abstract: how relevant is it, 0 to 4?",
  },
  "laya-score": {
    kind: "open, by Convai",
    text: "An open model, 421M parameters, run on our own machine. Asked the same question, word for word.",
  },
  "laya-typed-score": {
    kind: "open, by Convai",
    text: "The same Laya after its makers fine-tuned it for questions of this kind. Same question again.",
  },
  "cross-encoder": {
    kind: "the usual tool for this job",
    text: "A small model, 22M parameters, trained on real Bing searches. Here as the yardstick for the other three.",
    yardstick: true,
  },
};
const CARD_ORDER = ["jev-score", "laya-score", "laya-typed-score", "cross-encoder"];

/**
 * The experiment in three blocks: who the four are, what every one of them was
 * given, and how an order is graded, shown on the story's own question. It ends
 * on the number every re-ranker has to beat.
 */
export function RerankStoryExperiment({
  story,
  labels,
  questions,
  floor,
}: {
  story: StoryQuestion;
  labels: Record<string, string>;
  questions: number;
  /** The search's own nDCG@10, averaged over every question. */
  floor: number;
}) {
  const graded = [
    { label: "The search", sub: `right answer at ${pad(story.searchAt)}`, at: story.searchAt, value: story.searchNdcg },
    { label: story.helped.label, sub: `moved it to ${pad(story.helped.at)}`, at: story.helped.at, value: story.helped.ndcg },
    { label: story.buried.label, sub: `buried it at ${pad(story.buried.at)}`, at: story.buried.at, value: story.buried.ndcg },
  ];

  return (
    <section aria-labelledby="experiment" className="grid gap-10 border-t border-line pt-16 pb-12">
      <header className="grid max-w-[46rem] gap-3.5">
        <h2 id="experiment" className="text-balance text-h2 font-semibold leading-tight tracking-tight">
          The experiment: four re-rankers, <em>one test</em>.
        </h2>
        <p className="text-lead text-ink-soft">
          We gave four re-rankers the same job on the same questions, then graded every order they made.
        </p>
      </header>

      <div>
        <h3 className="mb-3.5 text-lead font-semibold">The four</h3>
        <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
          {CARD_ORDER.filter((m) => labels[m]).map((method) => (
            <div
              key={method}
              className={`grid content-start gap-1.5 border bg-surface px-4 py-3.5 ${
                WHO[method].yardstick ? "border-dashed border-line-strong" : "border-line"
              }`}
            >
              <b className="text-lg">{labels[method]}</b>
              <span className="text-micro text-ink-soft">{WHO[method].kind}</span>
              <p className="text-[0.9375rem] leading-normal">{WHO[method].text}</p>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="mb-3.5 text-lead font-semibold">The same test for all four</h3>
        <ol className="grid lg:grid-cols-4 lg:gap-4">
          {[
            [`${questions} health questions`, "From a public test set. For each one, the abstracts that answer it were marked in advance."],
            [`The search keeps ${story.results.length}`, `The same ${story.results.length} abstracts per question, for every re-ranker.`],
            ["Each re-ranker reorders them", "Only the reader changes. Nothing is tuned to these questions."],
            ["We grade the order", "Did the right answers move up?"],
          ].map(([head, body], i) => (
            <li
              key={head}
              className="grid grid-cols-[2.2rem_1fr] gap-1 border-t-[1.5px] border-ink py-3 lg:grid-cols-1"
            >
              <span className="numeric text-xs leading-7 text-ink-soft">{pad(i + 1)}</span>
              <b className="text-[1.05rem]">{head}</b>
              <p className="col-start-2 text-[0.9375rem] leading-snug text-ink-soft lg:col-start-1">{body}</p>
            </li>
          ))}
        </ol>
      </div>

      <div>
        <h3 className="mb-3.5 text-lead font-semibold">How an order is graded</h3>
        <p className="mb-4 max-w-[60ch] text-ink-soft">
          Only the top ten places count, and higher places are worth more. Here is &ldquo;{story.text}&rdquo; graded
          three ways.
        </p>
        <div className="grid gap-3.5 border border-line bg-surface p-4 sm:px-5.5 sm:py-5">
          <div className="hidden grid-cols-[8rem_minmax(0,1fr)_4rem] items-center gap-3 sm:grid" aria-hidden>
            <span className="text-[0.9375rem] text-ink-soft">Worth of each place</span>
            <Cells render={(place) => (place <= TOP ? placeWorth(place).toFixed(2).replace(/^0/, "") : place === 15 ? "0" : "")} />
            <span />
          </div>
          <p className="text-micro text-ink-soft sm:hidden">
            Place 1 is worth <span className="numeric">1.00</span>, place 7 <span className="numeric">0.33</span>, place
            10 <span className="numeric">0.29</span>. Below the line, nothing.
          </p>
          {graded.map((row) => (
            <div
              key={row.label}
              className="grid grid-cols-[minmax(0,1fr)_3.4rem] items-center gap-x-3 gap-y-1.5 border-t border-line pt-3 sm:grid-cols-[8rem_minmax(0,1fr)_4rem]"
            >
              <span className="col-span-2 text-[0.9375rem] leading-tight sm:col-span-1">
                {row.label}
                <small className="block text-micro text-ink-soft">{row.sub}</small>
              </span>
              <span role="img" aria-label={`${row.label}: the right answer at ${pad(row.at)}, graded ${row.value.toFixed(3)}`}>
                <Cells answer={row.at} />
              </span>
              <span className="numeric text-right text-lg font-medium">{row.value.toFixed(3)}</span>
            </div>
          ))}
          <p className="max-w-[64ch] border-t border-line pt-3 text-[0.9375rem] text-ink-soft">
            With one right answer, the grade is simply the worth of its place. With several, each counts by its place,
            and the best possible order still scores <b className="text-ink">1.000</b>. The measure is called{" "}
            <Term id="ndcg">nDCG@10</Term>.
          </p>
        </div>
      </div>

      <p className="max-w-[46rem] text-lead">
        Grade all {questions} questions and average. The search&apos;s own order, with no re-ranker at all, averages{" "}
        <span className="numeric font-medium">{floor.toFixed(3)}</span>. That is the number to beat.
      </p>
    </section>
  );
}

/** Twenty places with the top-ten line after the tenth: an answer's cell in violet, or each place's label. */
function Cells({ answer, render }: { answer?: number; render?: (place: number) => string }) {
  return (
    <span className="grid grid-cols-[repeat(10,minmax(0,1fr))_8px_repeat(10,minmax(0,1fr))] gap-[3px]">
      {Array.from({ length: 20 }, (_, i) => {
        const place = i + 1;
        const cell = render ? (
          <i key={place} className="numeric text-center text-[10px] not-italic text-ink-soft">
            {render(place)}
          </i>
        ) : (
          <i
            key={place}
            className={`h-[22px] ${place === answer ? "bg-mark" : place <= TOP ? "bg-line" : "bg-sunk"}`}
          />
        );
        return place === TOP + 1
          ? [<b key="line" className={`justify-self-center ${render ? "" : "w-[1.5px] bg-ink"}`} />, cell]
          : cell;
      })}
    </span>
  );
}
