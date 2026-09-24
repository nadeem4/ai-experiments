import results from "@/data/results.json";

export function SiteFooter() {
  return (
    <footer className="border-t border-line bg-surface">
      <div className="mx-auto grid max-w-[1100px] gap-2 px-4 py-8 text-micro text-ink-soft md:px-8">
        <p>
          Every number here is read out of a results file produced by{" "}
          <a href="https://github.com/nadeem4/ai-experiments" className="underline decoration-accent decoration-2 underline-offset-4 hover:text-ink">
            the code that ran the experiment
          </a>
          . Nothing on this site is illustrative.
        </p>
        <p className="numeric">
          Run finished {new Date(results.finished).toISOString().slice(0, 10)}
          {results.commit ? ` at commit ${results.commit}` : ""}, on {results.device.toUpperCase()}.
        </p>
      </div>
    </footer>
  );
}
