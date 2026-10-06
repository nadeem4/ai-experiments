import { notFound } from "next/navigation";
import { EXPERIMENTS, type ExperimentEntry } from "@/experiments.config";

export interface PublishedExperiment extends ExperimentEntry {
  href: string;
}

/** The experiments the site shows, in the config's order. */
export function published(): PublishedExperiment[] {
  return EXPERIMENTS.filter((e) => e.published).map((e) => ({ ...e, href: `/${e.slug}/` }));
}

/** An experiment the config does not name is not published: nothing goes out by default. */
export function isPublished(slug: string): boolean {
  return EXPERIMENTS.some((e) => e.slug === slug && e.published);
}

/** Called at the top of an experiment's page: renders the 404 unless it is published. */
export function requirePublished(slug: string): void {
  if (!isPublished(slug)) notFound();
}
