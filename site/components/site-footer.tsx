/**
 * The site now carries three experiments with three different runs, so the
 * footer states what is true of all of them and nothing that belongs to one.
 * Each page carries its own run's date, hardware and provenance.
 */
export function SiteFooter() {
  return (
    <footer className="border-t border-line bg-surface">
      <div className="mx-auto grid max-w-[1180px] gap-2 px-4 py-8 text-micro text-ink-soft md:px-8">
        <p>
          Every number here is read out of a results file produced by{" "}
          <a href="https://github.com/nadeem4/ai-experiments" className="underline decoration-line-strong underline-offset-4 hover:decoration-ink">
            the code that ran the experiment
          </a>
          , and every chart is one that experiment&apos;s report step drew from the same file. Nothing
          on this site is illustrative.
        </p>
      </div>
    </footer>
  );
}
