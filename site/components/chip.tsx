/** Where a number came from: metadata, so it is quiet and never emphasised. */
export function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="numeric ml-2 border border-line bg-sunk px-1.5 py-px align-middle text-[0.75em] font-normal text-ink-soft">
      {children}
    </span>
  );
}

/** An inline code span, in the mono face the rest of the evidence is set in. */
export function Mono({ children }: { children: React.ReactNode }) {
  return <code className="numeric bg-sunk px-1 text-[0.88em]">{children}</code>;
}

/**
 * One movement of the experiment. The sections are named for the job they do
 * rather than numbered, because the names are the sequence: a reader who lands
 * halfway down should be able to tell whether they are reading a prediction or
 * a result.
 */
export function Section({
  id,
  title,
  standfirst,
  children,
}: {
  id: string;
  title: string;
  standfirst?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-20 border-t border-line pt-8" aria-labelledby={id}>
      <h2 id={id} className="max-w-[24ch] text-h3 font-semibold leading-tight tracking-tight">
        {title}
      </h2>
      {standfirst && (
        <p className="mt-3 max-w-[62ch] text-body leading-relaxed text-ink-soft">{standfirst}</p>
      )}
      <div className="mt-8 grid min-w-0 gap-8">{children}</div>
    </section>
  );
}

/** The reading column: prose sits at a comfortable measure, charts do not. */
export function Prose({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`grid min-w-0 max-w-[62ch] gap-4 text-body leading-relaxed ${className}`}>{children}</div>
  );
}
