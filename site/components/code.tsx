"use client";

import { Check, Copy } from "@phosphor-icons/react";
import { useState } from "react";
import { tokenize, type Token } from "@/lib/highlight";

// The JSON follows the page's own rule: the only coloured token is the number,
// because the number is the thing the experiment measured.
const KIND: Record<Token["kind"], string> = {
  key: "text-ink-soft",
  string: "text-ink",
  number: "text-up",
  punct: "text-ink-soft/70",
};

/** Formatted JSON, coloured, with the played move marked. */
export function Code({ value, mark, className = "max-h-[18rem]" }: { value: unknown; mark?: string; className?: string }) {
  const json = JSON.stringify(value, null, 2);
  return (
    <pre className={`numeric overflow-auto bg-sunk p-4 text-micro leading-relaxed ${className}`}>
      <code>
        {tokenize(json, mark).map((t, i) => (
          <span key={i} className={`${KIND[t.kind]} ${t.hit ? "bg-mark-wash font-medium text-ink" : ""}`}>{t.text}</span>
        ))}
      </code>
    </pre>
  );
}

export function CopyButton({ value }: { value: unknown }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard?.writeText(JSON.stringify(value, null, 2)).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }).catch(() => {});
      }}
      className="numeric flex items-center gap-1.5 text-micro text-ink-soft hover:text-ink"
    >
      {copied ? <Check size={14} weight="bold" aria-hidden /> : <Copy size={14} weight="bold" aria-hidden />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}
