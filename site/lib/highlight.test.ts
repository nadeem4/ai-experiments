import { describe, expect, it } from "vitest";
import { tokenize } from "./highlight";

describe("tokenize", () => {
  it("labels keys, strings, numbers and punctuation", () => {
    const kinds = tokenize('{"a": "x", "b": 2}').map((t) => t.kind);
    expect(kinds).toEqual(["punct", "key", "punct", "string", "punct", "key", "punct", "number", "punct"]);
  });

  it("keeps the text so the output reads exactly like the input", () => {
    const json = '{\n  "state": {\n    "lane": "left"\n  }\n}';
    expect(tokenize(json).map((t) => t.text).join("")).toBe(json);
  });

  it("marks the lines that mention a highlighted value", () => {
    const tokens = tokenize('{"LANE_LEFT": 0.95, "IDLE": 0.05}', "LANE_LEFT");
    expect(tokens.find((t) => t.text.includes("LANE_LEFT"))?.hit).toBe(true);
    expect(tokens.find((t) => t.text.includes("IDLE"))?.hit).toBeFalsy();
  });
});
