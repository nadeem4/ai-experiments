export type TokenKind = "key" | "string" | "number" | "punct";
export interface Token { text: string; kind: TokenKind; hit?: boolean }

const PATTERN = /("(?:[^"\\]|\\.)*"\s*:)|("(?:[^"\\]|\\.)*")|(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi;

/**
 * Splits formatted JSON into coloured tokens, keeping every character so the
 * rendered output is identical to the input. `mark` flags tokens naming a value
 * worth pointing at, such as the option the model chose.
 */
export function tokenize(json: string, mark?: string): Token[] {
  const tokens: Token[] = [];
  let last = 0;
  for (const m of json.matchAll(PATTERN)) {
    const at = m.index!;
    if (at > last) tokens.push({ text: json.slice(last, at), kind: "punct" });
    const [text] = m;
    const kind: TokenKind = m[1] ? "key" : m[2] ? "string" : "number";
    tokens.push(mark && text.includes(mark) ? { text, kind, hit: true } : { text, kind });
    last = at + text.length;
  }
  if (last < json.length) tokens.push({ text: json.slice(last), kind: "punct" });
  return tokens;
}
