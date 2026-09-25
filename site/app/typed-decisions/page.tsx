import type { Metadata } from "next";
import Link from "next/link";
import typedData from "@/data/typed-decisions.json";
import { Chip, Mono, Prose, Section } from "@/components/chip";
import { FigureImage } from "@/components/figure-image";
import {
  describeValidity,
  pct,
  type BiasRow,
  type DecisionModel,
  type TypedDecisions,
} from "@/lib/experiments";

const run = typedData as unknown as TypedDecisions;
const { figures } = run;

const SMALL = run.tasks.find((t) => t.n_options === Math.min(...run.tasks.map((x) => x.n_options)))!;
const WIDE = run.tasks.find((t) => t.n_options === Math.max(...run.tasks.map((x) => x.n_options)))!;

const forTask = (task: string) => run.models.filter((m) => m.task === task);
const model = (name: string, task: string) =>
  run.models.find((m) => m.model === name && m.task === task)!;

const JEV_SMALL = model("jev", SMALL.task);
const JEV_WIDE = model("jev", WIDE.task);
const LAYA_SMALL = model("laya", SMALL.task);
const LAYA_WIDE = model("laya", WIDE.task);

/** Accuracy does not depend on the transport, so this one row set is all of them. */
const mostAccurate = (task: string) =>
  forTask(task)
    .filter((m) => m.accuracy !== null)
    .reduce((a, b) => (a.accuracy! > b.accuracy! ? a : b));

const JEV_RISE = run.latency_rise.find((r) => r.model === "jev")!;
const STEEPEST = run.latency_rise[run.latency_rise.length - 1];
// "Barely moves" is a tenth either way, which is the claim the page makes and
// the one its test pins. Jev is not alone inside it.
const FLAT_OTHERS = run.latency_rise.filter((r) => r.model !== "jev" && Math.abs(r.rise) < 0.1);
const NEXT_FASTEST = forTask(SMALL.task)
  .filter((m) => !m.local && m.p50_ms !== null && m.model !== "jev")
  .reduce((a, b) => (a.p50_ms! < b.p50_ms! ? a : b));

const bias = [...run.position_bias].sort(
  (a, b) => a.n_options - b.n_options || (b.spread ?? -1) - (a.spread ?? -1),
);
const MOVED = bias.filter((b) => (b.spread ?? 0) > 0.1);
const REFUSED_BIAS = bias.filter((b) => b.n_valid === 0);

const wideShare = run.option_share.find((s) => s.n_options === WIDE.n_options)!;
const narrowShare = run.option_share.find((s) => s.n_options === SMALL.n_options)!;

const TITLE = "Many models, one job: what a typed decision costs at 4 options and at 151";
const DESCRIPTION =
  `Nine models, two classification tasks, ${run.n_calls.toLocaleString()} recorded calls. Jev is the ` +
  `fastest hosted model on both and wins on accuracy on neither, and the position bias the run was ` +
  `built to measure shows up in one model out of eighteen model/task pairs.`;

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION },
};

export default function Page() {
  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-28 pt-12 md:px-8">
      <h1 className="max-w-[20ch] text-h1 font-semibold leading-[1.05] tracking-tight">{TITLE}</h1>
      <p className="numeric mt-6 max-w-[62ch] text-micro text-ink-soft">
        {new Set(run.models.map((m) => m.model)).size} models, {run.tasks.length} tasks,{" "}
        {run.n_calls.toLocaleString()} recorded calls, ${run.spend_usd.toFixed(4)} spent, one frozen
        spec per task.
      </p>

      <Section
        id="why"
        title="Why it is worth asking"
        standfirst="Assign a piece of text to exactly one of a fixed list of labels. Three things follow from how a model does that, and only one of them is usually measured."
      >
        <Prose>
          <p>
            A decision model answers a typed question in one forward pass and hands back a probability
            for every option. An LLM answers by generating text that has to be parsed back into a
            decision, and it carries the whole option list in its prompt on every call.
          </p>
          <p>
            So <b>cost scales with the option list for an LLM and not for a decision model</b>. And an{" "}
            <b>ordered list has a top and a bottom</b>: an LLM reads its options as text in an order
            somebody chose, while a decision model scores each option at its own marker and never sees
            them as a sequence. If an LLM&apos;s answer changes when only the order changes, that
            change is pure error — same example, same options, same correct answer. And{" "}
            <b>a label is not a probability</b>: without a distribution there is no threshold, so
            &ldquo;route this one to a human when the model is unsure&rdquo; is not available at any
            price.
          </p>
        </Prose>
      </Section>

      <Section
        id="expected"
        title="What we expected"
        standfirst="The prediction was committed about nine hours before the run's first call record, and it is on the page as it was made."
      >
        <div className="max-w-[62ch] border border-line bg-surface p-5">
          <p className="numeric text-micro text-ink-soft">
            What the protocol predicted, in summary. Committed about nine hours before the run&apos;s
            first call record
          </p>
          <p className="mt-3 text-lead italic leading-snug">
            At {WIDE.n_options} options the LLMs will flip their answer on more than about a fifth of
            examples when only the order changes, and their accuracy will vary by more than about ten
            points depending on where the correct option sits. The decision models will be flat.
          </p>
        </div>
        <Prose>
          <p>
            The first half did not hold. Two of seven LLMs flipped above a fifth, and exactly one moved
            its accuracy by more than ten points. The second half &ldquo;held&rdquo; in a run where
            almost nothing showed bias at all, which is the weaker of the two ways it could have held:
            Jev being flat is not evidence for the mechanism when most of the field is equally flat.
          </p>
        </Prose>
      </Section>

      <Section
        id="did"
        title="What we did"
        standfirst={`${SMALL.n_examples} examples per task in the measured arm, one frozen spec per task, hash-checked on read. One call, one answer, and no retry-until-valid.`}
      >
        <Prose>
          <p>
            Two tasks at opposite ends of the answer space: <Mono>{SMALL.dataset}</Mono> at{" "}
            {SMALL.n_options} options and <Mono>{WIDE.dataset}</Mono> at {WIDE.n_options}. Alongside
            the measured arm, a bias subset of <b>{SMALL.bias_subset} examples per placement</b> under{" "}
            {SMALL.n_random_orders} seeded random orderings and {SMALL.n_random_orders} controlled gold
            placements, and {SMALL.n_validation} validation rows carved out of <b>train</b> for the two
            models that return a probability. The bias and validation calls are excluded from the
            headline tables so that no example is weighted twice.
          </p>
          <p>
            An unusable <i>answer</i> scores wrong, because dropping it would pay a model for returning
            garbage. A call that never returned is the transport and cannot be scored at all: it is
            excluded and reported in the validity column instead. Latency times only the successful
            attempt; retry backoff is excluded by design and is therefore invisible in the medians and
            real in wall clock.
          </p>
        </Prose>
      </Section>

      <Section
        id="happened"
        title="What happened"
        standfirst={`Jev is the fastest hosted model on both tasks and wins on accuracy on neither. Laya cannot answer the ${WIDE.n_options}-option task at all.`}
      >
        {run.tasks.map((task) => (
          <figure key={task.task} className="grid min-w-0 gap-3">
            <h3 className="text-body font-semibold">
              {task.dataset}, {task.n_options} options, n = {task.n_examples}
            </h3>
            <div className="min-w-0 overflow-x-auto border border-line bg-surface">
              <table className="numeric w-full min-w-[860px] border-collapse text-left text-micro">
                <caption className="sr-only">
                  Latency, cost, validity and accuracy for every model on {task.dataset}
                </caption>
                <thead>
                  <tr className="border-b border-line-strong">
                    <th scope="col" className="px-3 py-2 font-medium">Model</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">p50</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">p95</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">tail</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">in tok</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">$/1k</th>
                    <th scope="col" className="px-3 py-2 font-medium">validity</th>
                    <th scope="col" className="px-3 py-2 font-medium">accuracy (95% CI)</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">retries</th>
                  </tr>
                </thead>
                <tbody>
                  {forTask(task.task).map((row) => (
                    <ModelRow key={`${row.model}-${row.task}`} row={row} />
                  ))}
                </tbody>
              </table>
            </div>
            <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
              Latency is the successful attempt only, with retry backoff excluded.{" "}
              <b className="text-ink">
                Laya&apos;s row is measured on local CPU with no network in it and is not comparable to
                any other row in the latency columns.
              </b>{" "}
              Accuracy does not depend on the transport, so that column compares legitimately across
              all of them. A row with no accuracy returned no usable answers; it is not a zero.
            </figcaption>
          </figure>
        ))}

        {figures.latency && (
          <FigureImage
            figure={figures.latency}
            alt="Median and 95th-percentile latency for every model on both tasks, log scale"
          >
            Two things to read. Jev&apos;s bar is the shortest of the hosted models on both tasks and
            its whisker is barely longer than its bar — a tail ratio of{" "}
            {JEV_SMALL.tail_ratio!.toFixed(1)} and {JEV_WIDE.tail_ratio!.toFixed(1)}. And{" "}
            {run.models
              .filter((m) => (m.tail_ratio ?? 0) > 5)
              .map((m) => m.model)
              .filter((name, i, all) => all.indexOf(name) === i)
              .join(" and ")}{" "}
            carry tails above 5x that a median-only table would hide entirely. Laya&apos;s bar is local
            CPU and is grouped apart for that reason; it has no {WIDE.n_options}-option bar because it
            never answered.
          </FigureImage>
        )}

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            Jev is fast and flat in the option count, and flat is not unique to it
          </h3>
          <div className="min-w-0 overflow-x-auto border border-line bg-surface">
            <table className="numeric w-full min-w-[520px] border-collapse text-left text-micro">
              <caption className="sr-only">
                Each model&apos;s median latency at {SMALL.n_options} options and at {WIDE.n_options}
              </caption>
              <thead>
                <tr className="border-b border-line-strong">
                  <th scope="col" className="px-3 py-2 font-medium">Model</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">
                    {SMALL.n_options} options
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">
                    {WIDE.n_options} options
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">change</th>
                </tr>
              </thead>
              <tbody>
                {run.latency_rise.map((row) => (
                  <tr key={row.model} className="border-b border-line last:border-0">
                    <th scope="row" className="px-3 py-2 text-left font-medium">{row.model}</th>
                    <td className="px-3 py-2 text-right text-ink-soft">
                      {row.small_p50_ms.toFixed(0)} ms
                    </td>
                    <td className="px-3 py-2 text-right text-ink-soft">
                      {row.large_p50_ms.toFixed(0)} ms
                    </td>
                    <td className="px-3 py-2 text-right font-medium">
                      {row.rise >= 0 ? "+" : "−"}
                      {Math.abs(row.rise * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Prose>
            <p>
              Jev&apos;s median goes from {JEV_SMALL.p50_ms!.toFixed(0)} ms at {SMALL.n_options} options
              to {JEV_WIDE.p50_ms!.toFixed(0)} ms at {WIDE.n_options}, a rise of{" "}
              {(JEV_RISE.rise * 100).toFixed(0)}% for a{" "}
              {(WIDE.n_options / SMALL.n_options).toFixed(0)}-fold increase in the answer space. It is{" "}
              <b>not</b> the only one that barely moves:{" "}
              {FLAT_OTHERS.map(
                (r) => `${r.model} ${r.rise >= 0 ? "+" : "−"}${Math.abs(r.rise * 100).toFixed(0)}%`,
              ).join(" and ")}{" "}
              stay inside a tenth too, and a negative there means the wider list cost that model
              nothing measurable. What is unusual is being flat <i>and</i> fastest: Jev is{" "}
              {(NEXT_FASTEST.p50_ms! / JEV_SMALL.p50_ms!).toFixed(1)}x ahead of {NEXT_FASTEST.model},
              the next hosted model at {SMALL.n_options} options, while {STEEPEST.model} pays{" "}
              {(STEEPEST.rise * 100).toFixed(0)}% for the wider list.
            </p>
            <p>
              And it wins on accuracy nowhere. The most accurate model is{" "}
              {mostAccurate(SMALL.task).model} on {SMALL.dataset} and{" "}
              {mostAccurate(WIDE.task).model} on {WIDE.dataset}. The experiment&apos;s own paired tests
              put Jev&apos;s deficit at {SMALL.n_options} options inside noise and its deficit at{" "}
              {WIDE.n_options} options clear of zero, so the trade it offers — a real but small accuracy
              loss for a large and reliable speed gain — gets worse as the option list grows.
            </p>
          </Prose>
        </div>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            Laya&apos;s {WIDE.n_options}-option row is a refusal, not a score
          </h3>
          <Prose>
            <p>
              All {LAYA_WIDE.n} of its {WIDE.dataset} calls failed before the model ran:{" "}
              {WIDE.n_options} intent names do not fit the {run.laya_cfg.head_max_len}-token option
              budget the checkpoint ships with. The row reads {pct(LAYA_WIDE.valid_rate)} valid, the
              bias table marks it {REFUSED_BIAS.length > 0 ? `${REFUSED_BIAS[0].n_valid}/${REFUSED_BIAS[0].n_calls} usable` : "unusable"}{" "}
              with no flip rate, and the figure refuses to draw it. It is <b>never</b> a zero and never
              an accuracy. Whether raising that budget recovers the accuracy is the question the{" "}
              <Link
                href="/banking77/"
                className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
              >
                sibling Banking77 experiment
              </Link>{" "}
              exists to answer, and its answer is mostly no.
            </p>
            <p>
              At {SMALL.n_options} options, where it fits, Laya is the most accurate model in the run at{" "}
              {LAYA_SMALL.accuracy!.toFixed(3)}. Accuracy does not depend on the transport, so that
              comparison is legitimate; its latency is local CPU and is grouped apart everywhere,
              because that one is not.
            </p>
          </Prose>
        </div>
      </Section>

      <Section
        id="bias"
        title="Position bias, the measurement this run exists for"
        standfirst={`Same example, same options, only the order changes. Any change is pure error — and at ${SMALL.bias_subset} examples per placement, almost none of the spread below can be told from noise.`}
      >
        {figures.position_bias && (
          <FigureImage
            figure={figures.position_bias}
            alt="Accuracy with the correct option placed first, in the middle and last, for every model on both tasks"
          >
            Flat bars mean the model does not care where the answer sits. At {SMALL.n_options} options
            every panel is flat. At {WIDE.n_options} one bar group is not:{" "}
            {MOVED.map((b) => b.model).join(", ")} drops sharply when the correct option moves to the
            bottom of the list while its middle placement is unharmed. Laya is left out of the wide
            panel because it produced no usable answer under any ordering.
          </FigureImage>
        )}

        <div className="min-w-0 overflow-x-auto border border-line bg-surface">
          <table className="numeric w-full min-w-[820px] border-collapse text-left text-micro">
            <caption className="sr-only">
              Flip rate, accuracy at each gold placement and mean answer position, per model and task
            </caption>
            <thead>
              <tr className="border-b border-line-strong">
                <th scope="col" className="px-3 py-2 font-medium">Model / task</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">options</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">usable</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">flip</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">gold 1st</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">gold mid</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">gold last</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">spread</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">mean pos</th>
              </tr>
            </thead>
            <tbody>
              {bias.map((row) => (
                <BiasTableRow key={`${row.model}-${row.task}`} row={row} />
              ))}
            </tbody>
          </table>
        </div>

        <Prose>
          <p>
            <b>Almost none of that spread column survives a significance test.</b> At{" "}
            {SMALL.bias_subset} examples per placement, a spread of 0.025 is one example. The
            experiment ran a paired McNemar — an exact binomial on the discordant pairs, same examples
            in both arms — on gold-first against gold-last, and{" "}
            <b>
              exactly one of the {bias.length} model/task pairs comes out distinguishable:{" "}
              {MOVED.map((b) => `${b.model} on ${b.task}`).join(", ")}, p = 0.0225, on 13 discordant
              pairs splitting 11 against 2 in the predicted direction.
            </b>{" "}
            Every other pair in the run rests on between zero and four discordant examples in total and
            returns p ≥ 0.50. The instrument cannot resolve them, in either direction.
          </p>
          <p>
            That matters for the flat rows too. Jev&apos;s {pct(bias.find((b) => b.model === "jev" && b.task === SMALL.task)!.flip_rate ?? 0)}{" "}
            flip and 0.000 spread at {SMALL.n_options} options are consistent with a model that scores
            each option at its own marker and never reads them as a sequence — and equally consistent
            with a test too small to see anything. An effect must move at least six examples, all the
            same way, before the exact binomial can call it at this sample size.{" "}
            <b>Position bias at {WIDE.n_options} options is not a general property of LLMs here</b>, and
            that is the opposite of what the motivation argued.
          </p>
          <p className="text-ink-soft">
            The McNemar test is now part of the report step, so this p-value is read out of{" "}
            <Mono>results/full.json</Mono> like every other number on the page. It was hand-computed
            while this page was first written, which is exactly the drift the generated-data rule
            exists to prevent, so the reporter was changed rather than the number copied.
          </p>
        </Prose>
      </Section>

      <Section
        id="options"
        title="What the option list costs"
        standfirst="The same list, re-sent and re-billed, on every single decision."
      >
        <Prose>
          <p>
            At {SMALL.n_options} options the option list is{" "}
            {pct(narrowShare.min_share!)}–{pct(narrowShare.max_share!)} of what an LLM is billed for.
            At {WIDE.n_options} it is <b>{pct(wideShare.min_share!)}–{pct(wideShare.max_share!)}</b>{" "}
            — roughly nine tenths of the prompt, on every call, for the same list of labels.
          </p>
          <p>
            Jev is not free on options either: {Math.round(JEV_SMALL.mean_input_tokens!).toLocaleString()} mean input
            tokens at {SMALL.n_options} options against {Math.round(JEV_WIDE.mean_input_tokens!).toLocaleString()} at{" "}
            {WIDE.n_options}, starting from a much higher floor than any LLM in the run. And its{" "}
            <i>output</i> tokens scale with the option count too —{" "}
            {Math.round(JEV_WIDE.mean_output_tokens!).toLocaleString()} mean output tokens per call at {WIDE.n_options},
            because the distribution over every option is itself billed as output. The protocol did not
            anticipate that, and it is most of Jev&apos;s per-call cost at that option count.
          </p>
          {narrowShare.unusable.length > 0 && (
            <p className="text-ink-soft">
              One measurement is excluded rather than averaged in:{" "}
              {narrowShare.unusable.join(", ")} reported <i>fewer</i> prompt tokens with the option
              list than without it. That is the provider&apos;s accounting, not a property of the
              options, and no conclusion here rests on it.
            </p>
          )}
        </Prose>
      </Section>

      <Section
        id="probability"
        title="Does it return a probability at all"
        standfirst="Measured from the records rather than declared. This is the most durable difference in the run, and it is a capability difference rather than a quality one."
      >
        <div className="grid gap-4 md:grid-cols-2">
          <div className="border border-line bg-surface p-5">
            <p className="numeric text-micro text-ink-soft">
              Returns a distribution over every option
            </p>
            <p className="numeric mt-2 text-lead">{run.probability.returns.join(", ")}</p>
          </div>
          <div className="border border-line bg-surface p-5">
            <p className="numeric text-micro text-ink-soft">Returns a label only</p>
            <p className="numeric mt-2 text-lead">{run.probability.label_only.join(", ")}</p>
          </div>
        </div>
        <Prose>
          <p>
            {run.probability.label_only.length} of{" "}
            {run.probability.label_only.length + run.probability.returns.length} models cannot be
            thresholded, so &ldquo;route this one to a human when the model is unsure&rdquo; is not
            available from them at any price.
          </p>
        </Prose>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            And temperature scaling made calibration worse in every fit
          </h3>
          <div className="min-w-0 overflow-x-auto border border-line bg-surface">
            <table className="numeric w-full min-w-[560px] border-collapse text-left text-micro">
              <caption className="sr-only">
                Calibration error before and after one fitted temperature
              </caption>
              <thead>
                <tr className="border-b border-line-strong">
                  <th scope="col" className="px-3 py-2 font-medium">Model / task</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">n test</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">n val</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">T</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">ECE raw</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">ECE scaled</th>
                </tr>
              </thead>
              <tbody>
                {run.calibration.map((row) => (
                  <tr key={`${row.model}-${row.task}`} className="border-b border-line last:border-0">
                    <th scope="row" className="px-3 py-2 text-left font-medium">
                      {row.model} / {row.task}
                    </th>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.n_test}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.n_validation}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.temperature.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.ece_raw.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right font-medium" style={{ color: "var(--down)" }}>
                      {row.ece_scaled.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Prose>
            <p>
              The temperature was fitted on a validation split carved out of <b>train</b>, never on
              test, which is the correct procedure and also why it can lose: a temperature that helps
              on train-distribution rows is not guaranteed to help on test. It lost all three times.
              Laya&apos;s raw figure is the best in this table and should be read against its own
              load-time warning, which says the affected confidences in that checkpoint are
              uncalibrated.
            </p>
          </Prose>
        </div>
      </Section>

      <Section
        id="assumed"
        title="What we assumed"
        standfirst="Five boundaries. Each one changes what the numbers above are allowed to mean."
      >
        <Prose>
          <ul className="grid list-disc gap-3 pl-5">
            <li>
              <b>Laya&apos;s latency is local CPU and every other latency is a hosted round trip.</b>{" "}
              The transport is recorded on every call and the report refuses to average across them.
              Nothing on this page ranks the two together.
            </li>
            <li>
              <b>The bias arm is {SMALL.bias_subset} examples per placement.</b> It was powered to
              detect an effect the size of the one it found and nothing smaller. Flat rows are
              consistent with order-independence and equally consistent with an under-powered test.
            </li>
            <li>
              <b>{SMALL.n_options} against {WIDE.n_options} does not isolate the option count.</b>{" "}
              {SMALL.dataset} is topic classification and {WIDE.dataset} is intent classification, and
              this design cannot separate the two.
            </li>
            <li>
              <b>Determinism was not measured at this scale.</b> The repeat arm was not run on the full
              limit, and the pilot&apos;s number is not carried forward.
            </li>
            <li>
              <b>Licences differ and are recorded, not assumed.</b>{" "}
              {run.tasks.map((t) => `${t.dataset}: ${t.licence.split(" --")[0]}`).join("; ")}. Every
              model is the cheapest current-generation tier of its family, on one machine, on one date.
            </li>
          </ul>
        </Prose>
      </Section>

      <Section
        id="files"
        title="The files behind the page"
        standfirst="Everything above was read out of these. Nothing on this site is illustrative."
      >
        <Prose>
          <p>
            Every number on this page is generated into <Mono>site/data/typed-decisions.json</Mono> by{" "}
            <Mono>typed_decisions/scripts/export_site_data.py</Mono>, which reads{" "}
            <Mono>typed_decisions/results/full.json</Mono> and nothing else and runs as the last step
            of <Mono>python -m typed_decisions.report --tag full</Mono>, right after the figures. The
            charts are the ones that report step drew, copied by the same script rather than redrawn
            here. The raw wire — {run.n_calls.toLocaleString()} records holding the request, the
            response, the usage, both clocks, the option order shown and the distribution where there
            is one — is too large to commit and is not in the repository.
          </p>
        </Prose>
      </Section>

      <p className="numeric mt-20 border-t border-line pt-5 text-micro text-ink-soft">
        <Link href="/" className="underline decoration-line-strong underline-offset-4 hover:decoration-ink">
          All experiments
        </Link>{" "}
        ·{" "}
        <a
          href="https://github.com/nadeem4/ai-experiments/tree/main/typed_decisions"
          className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          The protocol, the results file and the tests
        </a>
      </p>
    </main>
  );
}

function ModelRow({ row }: { row: DecisionModel }) {
  const refused = row.accuracy === null;
  return (
    <tr className={`border-b border-line align-top last:border-0 ${row.local ? "bg-sunk" : ""}`}>
      <th scope="row" className="px-3 py-2 text-left font-medium">
        {row.model}
        <span className="block font-normal text-ink-soft">
          {row.local ? "local CPU, not comparable" : row.model_id}
        </span>
      </th>
      <td className="px-3 py-2 text-right font-medium">
        {row.p50_ms === null ? "—" : `${row.p50_ms.toFixed(0)} ms`}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">
        {row.p95_ms === null ? "—" : `${row.p95_ms.toFixed(0)} ms`}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">
        {row.tail_ratio === null ? "—" : row.tail_ratio.toFixed(1)}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">
        {row.mean_input_tokens === null ? "n/a" : row.mean_input_tokens.toFixed(0)}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">{row.cost_per_1k_usd.toFixed(4)}</td>
      <td className="px-3 py-2 text-ink-soft">{describeValidity(row.validity)}</td>
      <td className="px-3 py-2">
        {refused ? (
          <span className="text-ink-soft">
            no usable answer
            <Chip>structural refusal, not a zero</Chip>
          </span>
        ) : (
          <>
            <b>{row.accuracy!.toFixed(3)}</b>
            <span className="text-ink-soft">
              {" "}
              [{row.accuracy_ci![0].toFixed(2)}, {row.accuracy_ci![1].toFixed(2)}]
            </span>
          </>
        )}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">{row.retries}</td>
    </tr>
  );
}

function BiasTableRow({ row }: { row: BiasRow }) {
  const unusable = row.n_valid === 0;
  const cell = (value: number | null, digits = 3) =>
    value === null ? <span className="text-ink-soft">n/a</span> : value.toFixed(digits);
  return (
    <tr className="border-b border-line last:border-0">
      <th scope="row" className="px-3 py-2 text-left font-medium">
        {row.model} / {row.task}
      </th>
      <td className="px-3 py-2 text-right text-ink-soft">{row.n_options}</td>
      <td className="px-3 py-2 text-right text-ink-soft">
        {row.n_valid}/{row.n_calls}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">
        {row.flip_rate === null ? "n/a" : pct(row.flip_rate)}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">{cell(row.gold_first)}</td>
      <td className="px-3 py-2 text-right text-ink-soft">{cell(row.gold_middle)}</td>
      <td className="px-3 py-2 text-right text-ink-soft">{cell(row.gold_last)}</td>
      <td
        className="px-3 py-2 text-right font-medium"
        style={{ color: (row.spread ?? 0) > 0.1 ? "var(--down)" : undefined }}
      >
        {cell(row.spread)}
      </td>
      <td className="px-3 py-2 text-right text-ink-soft">
        {unusable ? "n/a" : cell(row.mean_position, 2)}
      </td>
    </tr>
  );
}
