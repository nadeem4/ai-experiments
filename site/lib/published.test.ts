/**
 * What the site shows is decided in one place. An experiment left out of the
 * config, or marked unpublished, is in no menu, no index and at no URL.
 */
import { describe, expect, it } from "vitest";
import { EXPERIMENTS } from "@/experiments.config";
import { isPublished, published } from "@/lib/published";

describe("published", () => {
  it("lists only the experiments marked published, in the config's order", () => {
    const shown = published().map((e) => e.slug);
    expect(shown).toEqual(EXPERIMENTS.filter((e) => e.published).map((e) => e.slug));
  });

  it("gives each one a path the static export actually builds", () => {
    for (const e of published()) expect(e.href).toBe(`/${e.slug}/`);
  });
});

describe("isPublished", () => {
  it("is true for an experiment the config publishes", () => {
    const first = EXPERIMENTS.find((e) => e.published);
    if (first) expect(isPublished(first.slug)).toBe(true);
  });

  it("is false for one the config holds back", () => {
    const held = EXPERIMENTS.find((e) => !e.published);
    if (held) expect(isPublished(held.slug)).toBe(false);
  });

  it("is false for a slug the config does not know, rather than published by default", () => {
    expect(isPublished("no-such-experiment")).toBe(false);
  });
});
