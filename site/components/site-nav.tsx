import Link from "next/link";

const LINKS = [
  { href: "/", text: "Experiments" },
  { href: "/rerank/", text: "Re-ranker" },
];

export function SiteNav() {
  return (
    <header className="border-b border-line bg-surface">
      <nav aria-label="Site" className="mx-auto flex max-w-[1180px] flex-wrap items-baseline gap-x-6 gap-y-2 px-4 py-3 md:px-8">
        <Link href="/" className="font-semibold tracking-tight">
          AI Experiments
        </Link>
        <ul className="numeric flex flex-wrap gap-x-5 gap-y-1 text-micro text-ink-soft">
          {LINKS.map(({ href, text }) => (
            <li key={href}>
              <Link href={href} className="hover:text-ink">{text}</Link>
            </li>
          ))}
          <li>
            <a href="https://arena.codewithnk.com" className="hover:text-ink">Decision Arena</a>
          </li>
          <li>
            <a href="https://github.com/nadeem4/ai-experiments" className="hover:text-ink">GitHub</a>
          </li>
        </ul>
      </nav>
    </header>
  );
}
