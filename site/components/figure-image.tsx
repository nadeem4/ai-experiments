import type { Figure } from "@/lib/experiments";

/**
 * A chart the experiment's report step drew, shown exactly as it was written.
 * Nothing is redrawn in the browser: a figure built in a page takes its numbers
 * from somewhere other than the results file, and then drifts from the table
 * beside it the first time the run is redone.
 *
 * The plate stays white in both themes rather than being inverted or recoloured,
 * because the image is evidence and a filter would alter it.
 */
export function FigureImage({
  figure,
  alt,
  children,
}: {
  figure: Figure;
  alt: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="grid min-w-0 gap-3">
      <div className="min-w-0 overflow-x-auto border border-line bg-white p-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={figure.src}
          alt={alt}
          width={figure.width}
          height={figure.height}
          className="h-auto w-full min-w-[560px]"
        />
      </div>
      <figcaption className="max-w-[72ch] text-micro leading-relaxed text-ink-soft">
        {children}
      </figcaption>
    </figure>
  );
}
