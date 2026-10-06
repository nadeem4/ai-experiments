"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { matchQueries, scrollTopToShow, type RerankQuery } from "@/lib/rerank";

/**
 * Choosing which of the 323 questions to watch being re-ranked.
 *
 * Previous and next step through whatever the search has narrowed the list to,
 * so the two controls always agree with each other. The list itself is one tab
 * stop with the arrow keys moving inside it: 323 separate tab stops would make
 * the rest of the page unreachable by keyboard, which is the opposite of
 * accessible.
 */
export function RerankQueryPicker({
  queries,
  selected,
  onSelect,
}: {
  queries: RerankQuery[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const [term, setTerm] = useState("");
  const listId = useId();
  const searchId = useId();
  const listRef = useRef<HTMLUListElement>(null);
  const moveFocus = useRef(false);

  // The question on screen stays in the list whatever the search says, because
  // it can be chosen from the chart further down the page while a search is
  // still narrowing the list.
  const matches = useMemo(() => matchQueries(queries, term, selected), [queries, term, selected]);
  const at = matches.findIndex((q) => q.id === selected);

  // Keep the chosen row in view when it was chosen from somewhere else on the
  // page, such as a bar in the chart below.
  // Only the list scrolls, never the page.
  useEffect(() => {
    const list = listRef.current;
    const row = list?.querySelector<HTMLLIElement>(`[data-id="${CSS.escape(selected)}"]`);
    if (!list || !row) return;
    list.scrollTop = scrollTopToShow(
      { top: row.offsetTop, height: row.offsetHeight },
      { scrollTop: list.scrollTop, height: list.clientHeight },
    );
    if (moveFocus.current) {
      row.focus({ preventScroll: true });
      moveFocus.current = false;
    }
  }, [selected, matches]);

  const step = (by: number) => {
    if (!matches.length) return;
    const next = at < 0 ? 0 : Math.min(matches.length - 1, Math.max(0, at + by));
    onSelect(matches[next].id);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLUListElement>) => {
    const keys: Record<string, number | undefined> = { ArrowDown: 1, ArrowUp: -1 };
    const by = keys[event.key];
    if (by !== undefined) {
      event.preventDefault();
      moveFocus.current = true;
      step(by);
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      moveFocus.current = true;
      onSelect(matches[event.key === "Home" ? 0 : matches.length - 1].id);
    }
  };

  return (
    <div className="grid gap-3 border border-line bg-surface p-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-[12rem] flex-1">
          <label htmlFor={searchId} className="numeric block text-micro text-ink-soft">
            Search the {queries.length} questions
          </label>
          <input
            id={searchId}
            type="search"
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            placeholder="vitamin, PLAIN-1018, …"
            className="numeric mt-1 w-full border border-line bg-page px-2 py-1.5 text-micro text-ink placeholder:text-ink-soft"
          />
        </div>
        <div className="flex gap-2">
          <PickerButton onClick={() => step(-1)} disabled={at <= 0} label="Previous question">
            ◂ prev
          </PickerButton>
          <PickerButton
            onClick={() => step(1)}
            disabled={at < 0 || at >= matches.length - 1}
            label="Next question"
          >
            next ▸
          </PickerButton>
        </div>
      </div>

      <ul
        ref={listRef}
        id={listId}
        role="listbox"
        aria-label="Questions in the run"
        onKeyDown={onKeyDown}
        className="relative max-h-56 min-w-0 overflow-y-auto border border-line bg-page"
      >
        {matches.map((query) => {
          const open = query.id === selected;
          return (
            <li
              key={query.id}
              data-id={query.id}
              role="option"
              aria-selected={open}
              tabIndex={open ? 0 : -1}
              onClick={() => onSelect(query.id)}
              className={`flex cursor-pointer items-baseline gap-3 px-2 py-1 text-micro ${
                open ? "bg-sunk text-ink" : "text-ink-soft hover:bg-sunk"
              }`}
            >
              <span className="numeric shrink-0">{query.id}</span>
              <span className="min-w-0 flex-1 truncate">{query.text}</span>
              {!query.can_move && (
                <span className="numeric shrink-0 text-ink-soft">cannot move</span>
              )}
            </li>
          );
        })}
        {!matches.length && <li className="px-2 py-2 text-micro text-ink-soft">No question matches that.</li>}
      </ul>
    </div>
  );
}

function PickerButton({
  children,
  onClick,
  disabled,
  label,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className="numeric min-h-11 border border-line bg-page px-3 py-1.5 text-micro text-ink disabled:text-ink-soft disabled:opacity-50"
    >
      {children}
    </button>
  );
}
