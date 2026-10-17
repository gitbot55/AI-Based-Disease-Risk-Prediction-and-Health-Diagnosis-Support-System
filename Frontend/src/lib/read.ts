/**
 * Defensive readers for UNVERIFIED backend response shapes.
 * Each reader returns undefined when a field isn't present — never a fabricated value.
 */
import { SYMPTOMS, type SymptomKey } from "./assessment";

export type Obj = Record<string, unknown>;

export const isObj = (v: unknown): v is Obj => !!v && typeof v === "object" && !Array.isArray(v);

/** Unwrap common envelopes like { data: {...} } or { prediction: {...} }. */
export function unwrap(v: unknown, keys: string[] = ["data", "prediction", "result", "assessment"]): Obj {
  if (!isObj(v)) return {};
  for (const k of keys) if (isObj(v[k])) return { ...v, ...(v[k] as Obj) };
  return v;
}

export function pick<T = unknown>(o: Obj | undefined, keys: string[]): T | undefined {
  if (!o) return undefined;
  for (const k of keys) if (o[k] !== undefined && o[k] !== null) return o[k] as T;
  return undefined;
}

export const num = (o: Obj | undefined, keys: string[]) => {
  const v = pick(o, keys);
  const n = typeof v === "string" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : undefined;
};
export const str = (o: Obj | undefined, keys: string[]) => {
  const v = pick(o, keys);
  return typeof v === "string" || typeof v === "number" ? String(v) : undefined;
};

export const riskLevel = (o?: Obj) => str(o, ["risk_level", "risk_category", "category", "risk"]);
export const probability = (o?: Obj) => num(o, ["probability", "diabetes_probability", "prediction_probability"]);
export const riskScore = (o?: Obj) => num(o, ["risk_score", "score"]);
export const confidence = (o?: Obj) => num(o, ["confidence", "confidence_score"]);
export const recordId = (o?: Obj) => str(o, ["prediction_id", "id"]);
export const createdAt = (o?: Obj) => str(o, ["created_at", "timestamp", "date", "submitted_at"]);

export function recommendations(o?: Obj): string[] | undefined {
  const v = pick(o, ["recommendations", "advice", "suggestions"]);
  if (Array.isArray(v)) {
    const out = v
      .map((x) => (typeof x === "string" ? x : isObj(x) ? str(x, ["text", "message", "title", "recommendation"]) : undefined))
      .filter((x): x is string => !!x);
    return out.length ? out : undefined;
  }
  if (typeof v === "string" && v) return [v];
  return undefined;
}

export interface ShapContribution {
  feature: string;
  human_name: string;
  shap_value: number;
  direction: "positive" | "negative" | "neutral";
  impact: string;
}

export interface ShapExplanation {
  summary: string;
  feature_importance: ShapContribution[];
}

export function shapExplanation(o?: Obj): ShapExplanation | undefined {
  const m = pick<Obj>(o, ["metadata"]);
  if (!m) return undefined;
  const s = pick<Obj>(m, ["shap_explanation"]);
  if (!s) return undefined;
  const summary = str(s, ["summary"]);
  const fi = pick<unknown[]>(s, ["feature_importance"]);
  if (!summary || !Array.isArray(fi)) return undefined;
  
  const factors = fi.filter(isObj).map((f) => ({
    feature: str(f, ["feature"]) ?? "",
    human_name: str(f, ["human_name"]) ?? "",
    shap_value: num(f, ["shap_value"]) ?? 0,
    direction: (str(f, ["direction"]) as "positive" | "negative" | "neutral") ?? "neutral",
    impact: str(f, ["impact"]) ?? "",
  }));
  
  return { summary, feature_importance: factors };
}

/** Only returns durations if the backend genuinely returned symptom-day fields. */
export function persistedSymptoms(o?: Obj): Record<SymptomKey, number> | undefined {
  if (!o) return undefined;
  const sources = [o, isObj(o.symptoms) ? (o.symptoms as Obj) : undefined, isObj(o.symptom_durations) ? (o.symptom_durations as Obj) : undefined];
  for (const src of sources) {
    if (!src) continue;
    const out = {} as Record<SymptomKey, number>;
    let found = 0;
    for (const s of SYMPTOMS) {
      const v = src[s.key];
      if (typeof v === "number") {
        out[s.key] = v;
        found++;
      }
    }
    if (found === SYMPTOMS.length) return out;
  }
  return undefined;
}

export function formatPct(p: number) {
  const v = p <= 1 ? p * 100 : p;
  return `${v.toFixed(v < 10 ? 1 : 0)}%`;
}

export function formatDate(s?: string) {
  if (!s) return undefined;
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export type RiskTone = "low" | "mid" | "high" | "neutral";
export function riskTone(level?: string): RiskTone {
  if (!level) return "neutral";
  const l = level.toLowerCase();
  if (/(high|severe|elevated)/.test(l)) return "high";
  if (/(moderate|medium|mid)/.test(l)) return "mid";
  if (/(low|minimal|none)/.test(l)) return "low";
  return "neutral";
}

/** Find an array of rows inside a list response. */
export function listRows(v: unknown): Obj[] {
  if (Array.isArray(v)) return v.filter(isObj);
  if (!isObj(v)) return [];
  for (const k of ["predictions", "items", "data", "results", "records", "trends", "monthly", "history"]) {
    const x = v[k];
    if (Array.isArray(x)) return x.filter(isObj);
    if (isObj(x)) {
      const inner = listRows(x);
      if (inner.length) return inner;
    }
  }
  return [];
}

export function pagination(v: unknown) {
  const root = isObj(v) ? v : {};
  const p = isObj(root.pagination) ? (root.pagination as Obj) : isObj(root.data) && isObj((root.data as Obj).pagination) ? ((root.data as Obj).pagination as Obj) : root;
  return {
    page: num(p, ["page", "current_page"]),
    pages: num(p, ["pages", "total_pages"]),
    total: num(p, ["total", "total_items", "count"]),
  };
}
