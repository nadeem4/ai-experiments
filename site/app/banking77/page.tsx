import type { Metadata } from "next";
import Link from "next/link";
import bankingData from "@/data/banking77.json";
import { Chip, Mono, Prose, Section } from "@/components/chip";
import { FigureImage } from "@/components/figure-image";
import { pct, points, type Banking } from "@/lib/experiments";

const banking = bankingData as Banking;
const { headline, published, figures } = banking;

const arm = (name: string) => banking.arms.find((a) => a.arm === name)!;
const DEFAULT = arm(headline.default_arm);
const RAISED = arm(headline.raised_arm);
const WORKAROUND = arm(headline.workaround_arm);
const ENGLISH = arm("E-english");

const truncation = (name: string) => banking.truncation.find((t) => t.arm === name)!;
const DEFAULT_TRUNCATION = truncation(headline.default_arm);
const RAISED_TRUNCATION = truncation(headline.raised_arm);

const sweep = [...banking.option_sweep].sort((a, b) => a.n_options - b.n_options);
// The rungs where nothing truncates and nothing collides: the fall across those
// is the option count on its own, with the budget mechanism entirely absent.
const clean = sweep.filter((row) => row.identical_pairs === 0);
const cleanFall = (clean[0].accuracy - clean[clean.length - 1].accuracy) * 100;
const widestBudget = Math.max(...banking.arms.map((a) => a.head_max_len));

const TITLE = "Does Laya's Banking77 failure come from its token budget?";
const DESCRIPTION =
  `No. Giving all ${DEFAULT.n_options} options the room they need removes every truncation ` +
  `collision and buys ${points(headline.budget_gain_points)} accuracy points of a ` +
  `${headline.deficit_points.toFixed(2)}-point gap, and the documented workaround scores ` +
  `${points(headline.workaround_vs_default_points)} against doing nothing.`;

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION },
};

export default function Page() {
  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-28 pt-12 md:px-8">
      <h1 className="max-w-[19ch] text-h1 font-semibold leading-[1.05] tracking-tight">{TITLE}</h1>
      <p className="numeric mt-6 max-w-[62ch] text-micro text-ink-soft">
        BANKING77, all {DEFAULT.n_test.toLocaleString()} test rows, {DEFAULT.n_options} intents,{" "}
        {banking.arms.length} arms, inference only, every call on {banking.device.toUpperCase()}.
      </p>

      <Section
        id="why"
        title="Why it is worth asking"
        standfirst="Convai publishes a number that looks like a failure and an explanation for it. The explanation and the failure have opposite consequences for anyone choosing a model."
      >
        <Prose>
          <p>
            Laya scores each option at its own marker, and every option shares one{" "}
            <Mono>head_max_len</Mono> budget. At {DEFAULT.n_options} intents that leaves roughly four
            tokens each, and the model card blames exactly this for its published{" "}
            {published.laya.toFixed(3)} on BANKING77 against Jev&apos;s {published.jev.toFixed(3)}.
          </p>
          <p>
            If room alone recovers the accuracy, the published number is a configuration artifact and
            anyone deploying Laya on a wide label set should raise the budget, which costs nothing but
            context. If it does not, the number is a capability limit and the documented coarse-to-fine
            workaround is the only lever left. This experiment changes one thing at a time until the
            two can be told apart. Nothing here is trained.
          </p>
        </Prose>
      </Section>

      <Section
        id="expected"
        title="What we expected"
        standfirst="The hypothesis under test is Convai's, not ours, and the protocol wrote down what would falsify it before the sweep ran."
      >
        <div className="max-w-[62ch] border border-line bg-surface p-5">
          <p className="numeric text-micro text-ink-soft">
            The protocol&apos;s falsification condition, committed about twelve hours before the first
            call of this sweep
          </p>
          <p className="mt-3 text-lead italic leading-snug">
            Arms B-384, B-512 and B-768 failing to move accuracy above arm A&apos;s interval, while the
            option-count arms K-10, K-20 and K-40 do move it. That would mean accuracy tracks task
            difficulty rather than tokens per option, and the budget is not the explanation.
          </p>
        </div>
        <Prose>
          <p>
            It was not triggered, and the answer still came out against the hypothesis. The budget
            arms <i>did</i> move accuracy clear of arm A&apos;s interval — so the condition as written
            was not met — and the option-count arms moved it{" "}
            {(cleanFall / headline.budget_gain_points).toFixed(1)}x as far with the budget untouched.
            The mechanism is real and it is not the explanation.
          </p>
        </Prose>
      </Section>

      <Section
        id="did"
        title="What we did"
        standfirst={`One question, ${DEFAULT.n_options} options, ${DEFAULT.n_test.toLocaleString()} test rows in one seeded order shared by every arm. One thing varies per arm and nothing else.`}
      >
        <Prose>
          <p>
            <b>A-default</b> runs the shipped configuration and asks whether the published number
            reproduces. <b>B-384, B-512 and B-768</b> raise the budget and nothing else. <b>K-10, K-20
            and K-40</b> hold the budget still and narrow the label set, which separates &ldquo;tokens
            per option&rdquo; from &ldquo;this task is harder with more options&rdquo;.{" "}
            <b>{headline.workaround_arm}</b> is the workaround the card documents: a coarse group
            first, then the intent inside the group it chose. <b>E-english</b> swaps the checkpoint.
          </p>
          <p>
            The options are the {DEFAULT.n_options} intent labels with underscores replaced by spaces
            and nothing else, frozen to a file before anything ran. The one fitted quantity in the
            experiment is a temperature, fitted on a validation split carved out of train. The
            truncation diagnostics are read off the sequence Laya itself builds, not off a
            re-implementation of it.
          </p>
        </Prose>
      </Section>

      <Section
        id="happened"
        title="What happened"
        standfirst={`Room buys ${points(headline.budget_gain_points)} points of a ${headline.deficit_points.toFixed(2)}-point gap and then stops buying anything, and the workaround loses to doing nothing.`}
      >
        {figures.arms && (
          <FigureImage
            figure={figures.arms}
            alt={`Accuracy of every ${DEFAULT.n_options}-option arm, against the two published numbers`}
          >
            The whole verdict in one panel. Arm A lands on the published Laya figure, the raised-budget
            arms land a short step above it, and the published Jev line sits far to the right of all of
            them with nothing reaching for it. The two dashed lines are published numbers, not measured
            here — {published.note}.
          </FigureImage>
        )}

        <figure className="grid min-w-0 gap-3">
          <div className="min-w-0 overflow-x-auto border border-line bg-surface">
            <table className="numeric w-full min-w-[760px] border-collapse text-left text-micro">
              <caption className="sr-only">
                Accuracy, top-5, macro-F1, calibration error and latency for every arm
              </caption>
              <thead>
                <tr className="border-b border-line-strong">
                  <th scope="col" className="px-3 py-2 font-medium">Arm</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">n</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">options</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">budget</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">accuracy</th>
                  <th scope="col" className="px-3 py-2 font-medium">95% CI</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">top-5</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">macro-F1</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">ECE</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">p50</th>
                </tr>
              </thead>
              <tbody>
                {banking.arms.map((row) => (
                  <tr
                    key={row.arm}
                    className={`border-b border-line align-top last:border-0 ${
                      row.arm === headline.default_arm ? "bg-sunk" : ""
                    }`}
                  >
                    <th scope="row" className="px-3 py-2 text-left font-medium">
                      {row.arm}
                      {row.arm === headline.default_arm && (
                        <span className="block font-normal text-ink-soft">as shipped</span>
                      )}
                    </th>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.n_test.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.n_options}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.head_max_len}</td>
                    <td className="px-3 py-2 text-right font-medium">{row.accuracy.toFixed(4)}</td>
                    <td className="px-3 py-2 text-ink-soft">
                      [{row.ci95[0].toFixed(4)}, {row.ci95[1].toFixed(4)}]
                    </td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.top5.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.macro_f1.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.ece_raw.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.p50_ms.toFixed(0)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
            <b className="text-ink">
              Arm {headline.default_arm} reproduces the published number: {published.laya.toFixed(3)} sits inside
              its interval of [{DEFAULT.ci95[0].toFixed(4)}, {DEFAULT.ci95[1].toFixed(4)}].
            </b>{" "}
            The instrument is measuring what the card measured. The K arms score different examples
            from arm {headline.default_arm} and from each other, so their intervals are not a ranking
            against it and there is no paired test between them. Every latency is{" "}
            {banking.device.toUpperCase()} on a machine with no usable GPU; the card&apos;s published
            33 ms is a T4 figure and is not comparable to any number in this column.
          </figcaption>
        </figure>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            The budget is a cliff, not a dial: {headline.identical_budget_arms.join(", ")} agree to
            every decimal
          </h3>
          <Prose>
            <p>
              At {headline.saturates_at} tokens nothing truncates any more —{" "}
              {pct(DEFAULT_TRUNCATION.truncated_fraction, 1)} of options are cut at the shipped budget
              and {pct(RAISED_TRUNCATION.truncated_fraction, 0)} are cut above it — so the three raised
              arms build the same request and get the same answers back. They match on accuracy, top-5,
              macro-F1, calibration and the same paired split against arm {headline.default_arm}.
              Raising the budget past {headline.saturates_at} buys nothing because there is nothing
              left to buy. Their latencies differ by a few milliseconds, which is the machine, not the
              budget.
            </p>
          </Prose>
          {figures.budget_cliff && (
            <FigureImage
              figure={figures.budget_cliff}
              alt="Accuracy against the option budget: one step up at 384 and then flat"
            >
              One step and then a flat line. The point is the absence of a slope after{" "}
              {headline.saturates_at}: this is not a dial that keeps paying, and a straight line drawn
              through these four points would say something the run does not.
            </FigureImage>
          )}
        </div>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            Accuracy tracks the number of options, with the budget held still
          </h3>
          <div className="min-w-0 overflow-x-auto border border-line bg-surface">
            <table className="numeric w-full min-w-[620px] border-collapse text-left text-micro">
              <caption className="sr-only">
                Accuracy against the number of options, at the shipped budget
              </caption>
              <thead>
                <tr className="border-b border-line-strong">
                  <th scope="col" className="px-3 py-2 font-medium">Arm</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">options</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">tokens per option</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">identical pairs</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">accuracy</th>
                </tr>
              </thead>
              <tbody>
                {sweep.map((row) => (
                  <tr key={row.arm} className="border-b border-line last:border-0">
                    <th scope="row" className="px-3 py-2 text-left font-medium">{row.arm}</th>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.n_options}</td>
                    <td className="px-3 py-2 text-right text-ink-soft">
                      {row.tokens_per_option.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-right text-ink-soft">{row.identical_pairs}</td>
                    <td className="px-3 py-2 text-right font-medium">{row.accuracy.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Prose>
            <p>
              At 10, 20 and 40 options the budget mechanism is entirely absent: nothing truncates,
              nothing collides, and the room each option gets barely moves. Accuracy still falls by{" "}
              {cleanFall.toFixed(1)} points from {clean[0].n_options} to{" "}
              {clean[clean.length - 1].n_options} options. Whatever is degrading Laya as the label set
              widens is doing it while the budget
              is untouched. These arms score different examples from each other, so this is a pattern
              rather than a paired measurement, and the chance baseline moves with the option count
              too — 1 in 10 against 1 in {DEFAULT.n_options}.
            </p>
          </Prose>
          {figures.option_count && (
            <FigureImage
              figure={figures.option_count}
              alt="Accuracy against the number of options on the table, at a fixed budget"
            >
              A steady fall with the budget pinned at {DEFAULT_TRUNCATION.head_max_len}. Only the last
              point has any truncation in it at all, and the {clean.length} that do not still lose{" "}
              {cleanFall.toFixed(1)} points between them.
            </FigureImage>
          )}
        </div>

        <div className="grid min-w-0 gap-4">
          <h3 className="text-body font-semibold">
            The documented workaround is worse than doing nothing
          </h3>
          <Prose>
            <p>
              <b>
                {headline.workaround_arm} scores {WORKAROUND.accuracy.toFixed(4)}, which is{" "}
                {points(headline.workaround_vs_default_points)} points against changing nothing and{" "}
                {points(headline.workaround_vs_raised_points)} against raising the budget.
              </b>{" "}
              It is the arm with the <i>most</i> room per option of any {DEFAULT.n_options}-intent arm
              and has no truncation at either step. Its top-5 collapses to{" "}
              {WORKAROUND.top5.toFixed(4)}, the lowest of any arm, which is the tell: a wrong coarse
              commitment deletes the right intent from the candidate set and the second step cannot
              recover it. Giving Laya an easier question twice is worse than giving it the hard
              question once.
            </p>
          </Prose>
        </div>

        <figure className="grid min-w-0 gap-3">
          <div className="min-w-0 overflow-x-auto border border-line bg-surface">
            <table className="numeric w-full min-w-[680px] border-collapse text-left text-micro">
              <caption className="sr-only">
                McNemar tests between arm {headline.default_arm} and each other flat arm
              </caption>
              <thead>
                <tr className="border-b border-line-strong">
                  <th scope="col" className="px-3 py-2 font-medium">Paired comparison</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">n</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">difference</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">B better on</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">A better on</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">p</th>
                </tr>
              </thead>
              <tbody>
                {banking.mcnemar.map((row) => {
                  const delta = (row.accuracy_b - row.accuracy_a) * 100;
                  return (
                    <tr key={`${row.a}-${row.b}`} className="border-b border-line last:border-0">
                      <th scope="row" className="px-3 py-2 text-left font-medium">
                        {row.a} vs {row.b}
                      </th>
                      <td className="px-3 py-2 text-right text-ink-soft">
                        {row.n_paired.toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <b style={{ color: delta > 0 ? "var(--up)" : "var(--down)" }}>
                          {points(delta)}
                        </b>
                      </td>
                      <td className="px-3 py-2 text-right text-ink-soft">{row.n01}</td>
                      <td className="px-3 py-2 text-right text-ink-soft">{row.n10}</td>
                      <td className="px-3 py-2 text-right text-ink-soft">
                        {row.p_value.toExponential(1)}
                        <Chip>{row.p_value < 0.05 ? "separable" : "within noise"}</Chip>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
            Every comparison is McNemar over the same {DEFAULT.n_test.toLocaleString()} examples in
            both arms, so each one is a paired test rather than two intervals held up next to each
            other. All {banking.mcnemar.length} separate cleanly — including the two that go the wrong
            way.
          </figcaption>
        </figure>
      </Section>

      <Section
        id="mechanism"
        title="The mechanism, measured rather than argued about"
        standfirst="Two labels that truncate to the same string cannot be told apart by any amount of capability. That is the hypothesis stated as evidence, and it is countable."
      >
        <div className="min-w-0 overflow-x-auto border border-line bg-surface">
          <table className="numeric w-full min-w-[680px] border-collapse text-left text-micro">
            <caption className="sr-only">Truncation diagnostics for each flat arm</caption>
            <thead>
              <tr className="border-b border-line-strong">
                <th scope="col" className="px-3 py-2 font-medium">Configuration</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">budget</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">options</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">tokens per option</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">truncated</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">collapse onto another</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">identical pairs</th>
              </tr>
            </thead>
            <tbody>
              {banking.truncation.map((row) => (
                <tr key={row.arm} className="border-b border-line last:border-0">
                  <th scope="row" className="px-3 py-2 text-left font-medium">{row.arm}</th>
                  <td className="px-3 py-2 text-right text-ink-soft">{row.head_max_len}</td>
                  <td className="px-3 py-2 text-right text-ink-soft">{row.n_options}</td>
                  <td className="px-3 py-2 text-right text-ink-soft">
                    {row.mean_text_tokens_per_option.toFixed(2)}
                  </td>
                  <td className="px-3 py-2 text-right text-ink-soft">
                    {pct(row.truncated_fraction, 1)} ({row.truncated_options})
                  </td>
                  <td className="px-3 py-2 text-right text-ink-soft">{row.collided_options}</td>
                  <td className="px-3 py-2 text-right font-medium">{row.identical_pairs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Prose>
          <p>
            At the shipped budget {DEFAULT_TRUNCATION.collided_options} options collapse onto{" "}
            {Object.keys(banking.collisions).length} shared strings —{" "}
            {DEFAULT_TRUNCATION.identical_pairs} pairs the model cannot possibly separate. Decoded back
            out of the sequence it was given:
          </p>
        </Prose>
        <div className="min-w-0 max-w-[78ch] overflow-x-auto border border-line bg-sunk p-4">
          <dl className="numeric grid gap-3 text-micro">
            {Object.entries(banking.collisions).map(([truncated, labels]) => (
              <div key={truncated} className="grid gap-1">
                <dt className="font-medium">&ldquo;{truncated}&rdquo;</dt>
                {labels.map((original) => (
                  <dd key={original} className="pl-6 text-ink-soft">← {original}</dd>
                ))}
              </div>
            ))}
          </dl>
        </div>
        <Prose>
          <p>
            The English checkpoint is the same room, not less of it. Laya floors the per-option budget
            at four tokens, so {ENGLISH.head_max_len} and {DEFAULT.head_max_len} hand{" "}
            {DEFAULT.n_options} options the same allowance — the diagnostics agree to two decimals and
            produce the same {DEFAULT_TRUNCATION.identical_pairs} pairs. E-english still scores{" "}
            {ENGLISH.accuracy.toFixed(4)}, and its interval of [{ENGLISH.ci95[0].toFixed(4)},{" "}
            {ENGLISH.ci95[1].toFixed(4)}] does not contain the published {published.laya.toFixed(3)}. That gap is
            the checkpoint, not the budget.
          </p>
        </Prose>
      </Section>

      <Section
        id="assumed"
        title="What we assumed"
        standfirst="Five boundaries on the result. Each one changes what the numbers above are allowed to mean."
      >
        <Prose>
          <ul className="grid list-disc gap-3 pl-5">
            <li>
              <b>Jev is not measured here.</b> The {published.jev.toFixed(3)} is published — {published.note}. Every
              comparison against it on this page is a comparison to a published figure on a
              differently-scored task, not a measurement of two models against each other.
            </li>
            <li>
              <b>Every latency is {banking.device.toUpperCase()}</b>, on a machine with no usable GPU.
              Nothing in the latency column is comparable to a hosted API or to the card&apos;s T4
              figure.
            </li>
            <li>
              <b>The K arms do not hold the task fixed.</b> Narrowing the label set moves the chance
              baseline with it, so their fall mixes a capability limit with the task getting easier.
              They share no examples with arm {headline.default_arm} and carry no paired test.
            </li>
            <li>
              <b>The options are bare intent labels.</b> Hand-written descriptions would make every
              option longer and the truncation worse, so nothing here says what better option text
              would do, and nothing says what happens above {widestBudget} tokens of budget.
            </li>
            <li>
              <b>Some rows cannot move at all.</b> {DEFAULT_TRUNCATION.collided_options} of the{" "}
              {DEFAULT.n_options} options collapse onto {Object.keys(banking.collisions).length}{" "}
              shared strings at the shipped budget, so every test row whose gold label is one of them
              carries a label the model cannot name. How much of the{" "}
              {points(headline.budget_gain_points)} lands on those rows is computed from the run&apos;s
              wire log and is not in the committed results file, so this page does not carry it — the
              experiment&apos;s own write-up says the reporter should emit it.
            </li>
          </ul>
        </Prose>
      </Section>

      <Section
        id="understood"
        title="What we understood"
        standfirst="The hypothesis survives as a mechanism and fails as an explanation."
      >
        <Prose>
          <p>
            <b>The token budget is a real effect and a minor one.</b> Removing every truncation
            collision moves accuracy from {DEFAULT.accuracy.toFixed(4)} to{" "}
            {RAISED.accuracy.toFixed(4)} — {points(headline.budget_gain_points)} points, separable at
            p&nbsp;={" "}
            {banking.mcnemar
              .find((m) => m.b === headline.raised_arm)!
              .p_value.toExponential(1)}{" "}
            — which is {pct(headline.share_of_deficit, 1)} of the{" "}
            {headline.deficit_points.toFixed(2)}-point gap to the published Jev number, about a ninth
            of it. {headline.remaining_points.toFixed(1)} points are still there with every option
            fully spelled out. Convai&apos;s explanation is supported as a small effect and not
            supported as an explanation.
          </p>
          <p>
            <b>The workaround is not a workaround.</b> Coarse-to-fine exists in the card as the thing
            to do when the label set is too wide. It is the worst {DEFAULT.n_options}-intent arm in
            the sweep, {points(headline.workaround_vs_default_points)} points against doing nothing at
            all. A negative result is still a result, and this page is where it stays.
          </p>
          <p>
            <b>What the other {headline.remaining_points.toFixed(0)} points are is not answered here.</b>{" "}
            This experiment is inference-only. It can say the budget is not the explanation; it cannot
            say whether the remainder is training data, capacity, or the masked-scoring head.
            &ldquo;A training problem&rdquo; is a hypothesis this run leaves standing, not one it
            establishes.
          </p>
        </Prose>
      </Section>

      <Section
        id="files"
        title="The files behind the page"
        standfirst="Everything above was read out of these. Nothing on this site is illustrative."
      >
        <Prose>
          <p>
            Every number on this page is generated into{" "}
            <Mono>site/data/banking77.json</Mono> by{" "}
            <Mono>banking77/scripts/export_site_data.py</Mono>, which reads{" "}
            <Mono>banking77/results/full.json</Mono> and nothing else and runs as the last step of{" "}
            <Mono>python -m banking77.report --tag full</Mono>. The charts are the ones the report
            drew, copied by the same script. The run&apos;s wire log holds one record per arm and
            example with the exact request, the exact response and the full{" "}
            {DEFAULT.n_options}-way distribution; it is too large to commit and is not in the
            repository.
          </p>
        </Prose>
      </Section>

      <p className="numeric mt-20 border-t border-line pt-5 text-micro text-ink-soft">
        <Link href="/" className="underline decoration-line-strong underline-offset-4 hover:decoration-ink">
          All experiments
        </Link>{" "}
        ·{" "}
        <a
          href="https://github.com/nadeem4/ai-experiments/tree/main/banking77"
          className="underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          The protocol, the results file and the tests
        </a>
      </p>
    </main>
  );
}
