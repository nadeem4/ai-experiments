"use client";

import { useEffect, useId, useRef, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { ArrowUpRightIcon, XIcon } from "@phosphor-icons/react";
import { GLOSSARY, type TermId } from "@/lib/rerank-glossary";

/**
 * A word on the page that explains itself.
 *
 * Built on the browser's popover: one card open at a time, a click or tap
 * outside or Escape closes it, and focus returns to the word. Wide screens
 * anchor the card under the word; narrow ones raise it as a sheet from the
 * bottom, where a small anchored card would crowd the text it explains. A card
 * never holds another term, so explanations never open other explanations.
 *
 * The card is portalled into <body>: the word sits inside a paragraph, and a
 * paragraph cannot hold the card's block content.
 */
export function Term({ id, children }: { id: TermId; children?: React.ReactNode }) {
  const entry = GLOSSARY[id];
  const cardId = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const card = useRef<HTMLDivElement>(null);
  // True only in the browser, so the server never renders a portal it cannot place.
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );

  useEffect(() => {
    const node = card.current;
    if (!node) return;
    // Placed when it opens rather than tracked: absolute in the top layer means
    // the card scrolls with the page, staying beside its word without a listener.
    const place = (event: Event) => {
      if ((event as ToggleEvent).newState !== "open" || !trigger.current) return;
      if (window.matchMedia("(max-width: 639px)").matches) {
        node.style.removeProperty("top");
        node.style.removeProperty("left");
        return;
      }
      const word = trigger.current.getBoundingClientRect();
      const width = node.offsetWidth;
      const height = node.offsetHeight;
      const left = Math.min(
        Math.max(16, word.left + word.width / 2 - width / 2),
        window.innerWidth - width - 16,
      );
      const below = word.bottom + 10;
      const top = below + height > window.innerHeight - 16 ? word.top - 10 - height : below;
      node.style.left = `${left + window.scrollX}px`;
      node.style.top = `${top + window.scrollY}px`;
    };
    node.addEventListener("toggle", place);
    return () => node.removeEventListener("toggle", place);
  }, [mounted]);

  return (
    <>
      <button
        ref={trigger}
        type="button"
        popoverTarget={cardId}
        className="term-trigger"
        aria-label={`${children ? String(children) : entry.term}: what this means`}
      >
        {children ?? entry.term}
      </button>
      {mounted &&
        createPortal(
          <div
            ref={card}
            id={cardId}
            popover="auto"
            className="term-card"
            role="dialog"
            aria-label={entry.term}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-lead font-semibold leading-tight text-ink">{entry.term}</p>
                <p className="numeric mt-1 text-micro text-ink-soft">{entry.kind}</p>
              </div>
              <button
                type="button"
                popoverTarget={cardId}
                popoverTargetAction="hide"
                className="term-close"
                aria-label="Close"
              >
                <XIcon size={18} weight="bold" aria-hidden />
              </button>
            </div>
            <p className="mt-3 text-body leading-relaxed text-ink">{entry.says}</p>
            {entry.here && (
              <p className="numeric mt-3 border-t border-line pt-3 text-micro leading-relaxed text-ink-soft">
                <span className="text-ink">In this run.</span> {entry.here}
              </p>
            )}
            {entry.link && (
              <a
                href={entry.link.href}
                target="_blank"
                rel="noreferrer"
                className="mt-4 inline-flex items-center gap-1.5 text-micro text-ink underline decoration-line-strong underline-offset-4 hover:decoration-ink"
              >
                {entry.link.label}
                <ArrowUpRightIcon size={14} weight="bold" aria-hidden />
              </a>
            )}
          </div>,
          document.body,
        )}
    </>
  );
}
