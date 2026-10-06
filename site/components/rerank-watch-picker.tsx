"use client";

import { useEffect, useId, useRef, useState } from "react";
import { CaretDownIcon } from "@phosphor-icons/react";
import { matchQueries, type RerankQuery } from "@/lib/rerank";

// Four questions that each show something different: two re-rankers agree, all
// of them help, only one finds the answer, and none can because nothing is there.
const QUICK = ["PLAIN-2081", "PLAIN-2680", "PLAIN-1679", "PLAIN-1008"];
const MAX_RESULTS = 60;

const answerCount = (q: RerankQuery) =>
  q.relevant.length ? `${q.relevant.length} answer${q.relevant.length === 1 ? "" : "s"}` : "no answer";

/**
 * Choosing one of the 323 questions. It looks and opens like a dropdown: a
 * labelled field with the current question and a caret, and a list under it,
 * with a search box on top because 323 entries are too many to scroll. On
 * phones the list rises as a bottom sheet (the CSS decides). The browser's
 * popover closes it on Escape or a tap outside.
 */
export function RerankWatchPicker({
  queries,
  selected,
  onSelect,
  sheetId,
}: {
  queries: RerankQuery[];
  selected: RerankQuery;
  onSelect: (id: string) => void;
  /** Shared with the phone's sticky bar, whose Change button opens the same sheet. */
  sheetId: string;
}) {
  const searchId = useId();
  const labelId = useId();
  const valueId = useId();
  const sheet = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const [term, setTerm] = useState("");

  useEffect(() => {
    const node = sheet.current;
    if (!node) return;
    const opened = (event: Event) => {
      if ((event as ToggleEvent).newState !== "open") return;
      if (window.matchMedia("(min-width: 1024px)").matches && trigger.current) {
        const r = trigger.current.getBoundingClientRect();
        node.style.left = `${r.left + window.scrollX}px`;
        node.style.top = `${r.bottom + 6 + window.scrollY}px`;
      } else {
        node.style.removeProperty("left");
        node.style.removeProperty("top");
      }
      // Open at the current question, as a dropdown does. The list scrolls
      // itself; scrollIntoView would scroll the page as well.
      const list = node.querySelector("ul");
      const current = list?.querySelector<HTMLElement>('[aria-current="true"]');
      if (list && current) list.scrollTop = current.offsetTop - list.clientHeight / 2 + current.offsetHeight / 2;
      search.current?.focus({ preventScroll: true });
    };
    node.addEventListener("toggle", opened);
    return () => node.removeEventListener("toggle", opened);
  }, []);

  const choose = (id: string) => {
    sheet.current?.hidePopover();
    setTerm("");
    if (id !== selected.id) onSelect(id);
  };

  const hits = matchQueries(queries, term);
  // The cap only applies to a search; with nothing typed the list is all of
  // them, so the current question is always there to open at.
  const shown = term.trim() ? hits.slice(0, MAX_RESULTS) : hits;
  const quick = QUICK.map((id) => queries.find((q) => q.id === id)).filter((q): q is RerankQuery => Boolean(q));

  const onListKey = (event: React.KeyboardEvent<HTMLUListElement>) => {
    const items = [...event.currentTarget.querySelectorAll("button")];
    const at = items.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "ArrowDown" && at < items.length - 1) {
      event.preventDefault();
      items[at + 1].focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      (at > 0 ? items[at - 1] : search.current)?.focus();
    }
  };

  return (
    <>
      <div className="grid gap-1.5">
        <span id={labelId} className="text-body font-semibold">
          Select a question
        </span>
        <button
          ref={trigger}
          type="button"
          popoverTarget={sheetId}
          aria-haspopup="listbox"
          aria-labelledby={`${labelId} ${valueId}`}
          className="grid min-h-12 w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border border-line-strong bg-surface px-3.5 py-2.5 text-left hover:border-ink"
        >
          <span id={valueId} className="text-body leading-snug">
            {selected.text}
          </span>
          <CaretDownIcon size={18} weight="bold" aria-hidden className="text-ink-soft" />
        </button>
      </div>

      <div ref={sheet} id={sheetId} popover="auto" className="watch-sheet" role="dialog" aria-label="Choose a question">
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 px-4 pb-2 pt-4">
          <label htmlFor={searchId} className="sr-only">
            Search the {queries.length} questions
          </label>
          <input
            ref={search}
            id={searchId}
            type="search"
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && shown[0]) choose(shown[0].id);
              if (event.key === "ArrowDown") {
                event.preventDefault();
                (sheet.current?.querySelector("ul button") as HTMLButtonElement | null)?.focus();
              }
            }}
            placeholder={`Search ${queries.length} questions`}
            autoComplete="off"
            className="min-h-11 border border-line-strong bg-page px-3 text-base"
          />
          <button
            type="button"
            popoverTarget={sheetId}
            popoverTargetAction="hide"
            className="min-h-11 border border-line bg-page px-3 text-micro"
          >
            Close
          </button>
        </div>
        <div className="flex flex-wrap gap-1.5 px-4 pb-3" aria-label="Places to start">
          {quick.map((q) => (
            <button key={q.id} type="button" onClick={() => choose(q.id)} className="min-h-10 border border-line bg-page px-2.5 text-micro">
              {q.text}
            </button>
          ))}
        </div>
        <ul className="relative overflow-y-auto border-t border-line pb-2" aria-label="Questions" onKeyDown={onListKey}>
          {shown.map((q) => (
            <li key={q.id}>
              <button
                type="button"
                onClick={() => choose(q.id)}
                aria-current={q.id === selected.id}
                className={`grid min-h-12 w-full grid-cols-[minmax(0,1fr)_auto] items-baseline gap-3 px-4 py-2.5 text-left text-body leading-snug hover:bg-sunk focus-visible:bg-sunk ${
                  q.id === selected.id ? "bg-mark-wash" : ""
                }`}
              >
                {q.text}
                <span className="numeric text-micro text-ink-soft">{answerCount(q)}</span>
              </button>
            </li>
          ))}
          {shown.length < hits.length && (
            <li className="px-4 py-2 text-micro text-ink-soft">
              Showing {MAX_RESULTS} of {hits.length}. Type to narrow the list.
            </li>
          )}
          {hits.length === 0 && (
            <li className="px-4 py-2 text-micro text-ink-soft">
              No question matches &ldquo;{term}&rdquo;. Try one word, such as vitamin.
            </li>
          )}
        </ul>
      </div>
    </>
  );
}
