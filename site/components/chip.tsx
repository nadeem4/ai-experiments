/** Where a number came from: metadata, so it is quiet and never emphasised. */
export function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="ml-2 rounded-sm border border-line bg-sunk px-1.5 py-px align-middle font-mono text-micro font-normal text-ink-soft">
      {children}
    </span>
  );
}

/** An inline code span, in the site's mono face. */
export function Mono({ children }: { children: React.ReactNode }) {
  return <code className="rounded-sm bg-sunk px-1 font-mono text-[0.9em]">{children}</code>;
}

export function Section({ id, n, title, standfirst, children }: {
  id: string; n: number; title: string; standfirst?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="mt-16 border-t-2 border-line pt-6" aria-labelledby={id}>
      <div className="flex items-baseline gap-3">
        <span aria-hidden className="numeric text-h2 font-extrabold leading-none text-line-strong">{n}</span>
        <h2 id={id} className="max-w-[24ch] text-h3 font-extrabold tracking-tight">{title}</h2>
      </div>
      {standfirst && <p className="mt-3 max-w-[70ch] text-body leading-relaxed text-ink-soft">{standfirst}</p>}
      <div className="mt-6 grid gap-6">{children}</div>
    </section>
  );
}
