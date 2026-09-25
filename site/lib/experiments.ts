/**
 * The two generated data files, read as `scripts/export_site_data.py` writes them.
 *
 * Nothing here recomputes a result. The comparisons the pages make were done in
 * the experiments, against their own results files; these are the types that
 * carry them and the three formatters the pages share.
 */

export interface Figure {
  src: string;
  width: number;
  height: number;
}

// --- banking77 --------------------------------------------------------------

export interface BankingArm {
  arm: string;
  n_test: number;
  n_options: number;
  head_max_len: number;
  checkpoint: string;
  accuracy: number;
  ci95: number[];
  top5: number;
  macro_f1: number;
  ece_raw: number;
  temperature: number | null;
  ece_after_temperature: number | null;
  p50_ms: number;
  p95_ms: number;
}

export interface BankingHeadline {
  default_arm: string;
  raised_arm: string;
  workaround_arm: string;
  deficit_points: number;
  budget_gain_points: number;
  share_of_deficit: number;
  remaining_points: number;
  workaround_vs_default_points: number;
  workaround_vs_raised_points: number;
  identical_budget_arms: string[];
  saturates_at: number;
}

export interface Banking {
  tag: string;
  device: string;
  published: { source: string; laya: number; jev: number; note: string };
  arms: BankingArm[];
  option_sweep: {
    arm: string;
    n_options: number;
    tokens_per_option: number;
    identical_pairs: number;
    accuracy: number;
    accuracy_ci95: number[];
  }[];
  mcnemar: {
    a: string;
    b: string;
    n_paired: number;
    accuracy_a: number;
    accuracy_b: number;
    n10: number;
    n01: number;
    p_value: number;
  }[];
  truncation: {
    arm: string;
    head_max_len: number;
    n_options: number;
    mean_text_tokens_per_option: number;
    truncated_fraction: number;
    truncated_options: number;
    collided_options: number;
    identical_pairs: number;
  }[];
  collisions: Record<string, string[]>;
  headline: BankingHeadline;
  figures: Record<string, Figure>;
}

// --- typed_decisions --------------------------------------------------------

export interface DecisionModel {
  model: string;
  model_id: string;
  task: string;
  n_options: number;
  transport: string;
  local: boolean;
  n: number;
  p50_ms: number | null;
  p95_ms: number | null;
  tail_ratio: number | null;
  accuracy: number | null;
  accuracy_ci: number[] | null;
  valid_rate: number;
  validity: Record<string, number>;
  retries: number;
  cost_per_1k_usd: number;
  cost_per_correct_usd: number | null;
  mean_input_tokens: number | null;
  mean_output_tokens: number | null;
  returns_probability: boolean;
}

export interface BiasRow {
  model: string;
  task: string;
  n_options: number;
  n_examples: number;
  n_calls: number;
  n_valid: number;
  flip_rate: number | null;
  gold_first: number | null;
  gold_middle: number | null;
  gold_last: number | null;
  spread: number | null;
  mean_position: number | null;
}

export interface TypedDecisions {
  tag: string;
  written_at: string;
  n_calls: number;
  spend_usd: number;
  tasks: {
    task: string;
    dataset: string;
    licence: string;
    n_examples: number;
    n_options: number;
    n_validation: number;
    bias_subset: number;
    n_random_orders: number;
  }[];
  models: DecisionModel[];
  latency_rise: { model: string; small_p50_ms: number; large_p50_ms: number; rise: number }[];
  position_bias: BiasRow[];
  option_share: {
    task: string;
    n_options: number;
    n_models: number;
    min_share: number | null;
    max_share: number | null;
    unusable: string[];
  }[];
  calibration: {
    model: string;
    task: string;
    temperature: number;
    ece_raw: number;
    ece_scaled: number;
    n_test: number;
    n_validation: number;
  }[];
  probability: { returns: string[]; label_only: string[] };
  laya_cfg: { head_max_len: number; max_len: number };
  figures: Record<string, Figure>;
}

// --- the formatters the two pages share --------------------------------------

export const pct = (n: number, digits = 0) => `${(n * 100).toFixed(digits)}%`;

/** A difference in accuracy points, with the sign kept: a loss reads as a loss. */
export const points = (n: number, digits = 2) =>
  `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;

/**
 * Validity by kind, valid first. A model that failed 24 of its calls and a model
 * that answered 276 questions wrongly are two different facts, and a bare count
 * of correct answers cannot tell them apart.
 */
export function describeValidity(validity: Record<string, number>): string {
  const entries = Object.entries(validity).sort(
    ([a], [b]) => Number(b === "valid") - Number(a === "valid"),
  );
  return entries.map(([kind, n]) => `${n} ${kind}`).join(", ");
}
