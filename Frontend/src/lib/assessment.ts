/** Symptom-first assessment model and payload builder (pure, testable). */

export const SYMPTOMS = [
  { key: "frequent_urination_days", label: "Frequent urination", hint: "Needing to urinate more often than usual, including at night." },
  { key: "excessive_thirst_days", label: "Excessive thirst", hint: "Feeling unusually thirsty even after drinking." },
  { key: "fatigue_days", label: "Fatigue or unusual tiredness", hint: "Tiredness that feels out of the ordinary for you." },
  { key: "blurred_vision_days", label: "Blurred vision", hint: "Vision that seems less sharp than normal." },
  { key: "slow_healing_wounds_days", label: "Slow-healing wounds", hint: "Cuts or sores that take longer than usual to heal." },
  { key: "weight_loss_days", label: "Unexplained weight loss", hint: "Losing weight without trying to." },
] as const;

export type SymptomKey = (typeof SYMPTOMS)[number]["key"];

export const MIN_DAYS = 1;
export const MAX_DAYS = 30;
export const MIN_AGE = 1;
export const MAX_AGE = 120;

export interface SymptomAnswer {
  present: boolean | null; // null = not answered yet
  days: number | null;
}

export interface AssessmentForm {
  age: string;
  sex: string;
  symptoms: Record<SymptomKey, SymptomAnswer>;
  lifestyle: { smoking: boolean | null; alcohol: boolean | null; exercise: boolean | null };
}

export interface AssessmentPayload {
  age: number;
  sex: number;
  symptoms: Record<SymptomKey, number>;
  lifestyle: { smoking: boolean; alcohol: boolean; exercise: boolean };
}

export function emptyForm(): AssessmentForm {
  const symptoms = {} as Record<SymptomKey, SymptomAnswer>;
  for (const s of SYMPTOMS) symptoms[s.key] = { present: null, days: null };
  return { age: "", sex: "", symptoms, lifestyle: { smoking: null, alcohol: null, exercise: null } };
}

export function validateAge(age: string): string | null {
  if (age.trim() === "") return "Please enter your age.";
  if (!/^\d+$/.test(age.trim())) return "Age must be a whole number.";
  const n = Number(age);
  if (n < MIN_AGE || n > MAX_AGE) return `Age must be between ${MIN_AGE} and ${MAX_AGE}.`;
  return null;
}

export function validateSex(sex: string): string | null {
  if (sex !== "1" && sex !== "0") return "Please select your biological sex.";
  return null;
}

export function validateSymptom(a: SymptomAnswer): string | null {
  if (a.present === null) return "Please choose an option.";
  if (a.present) {
    if (a.days === null || Number.isNaN(a.days)) return "Please enter how many days.";
    if (!Number.isInteger(a.days)) return "Days must be a whole number.";
    if (a.days < MIN_DAYS || a.days > MAX_DAYS) return `Enter between ${MIN_DAYS} and ${MAX_DAYS} days.`;
  }
  return null;
}

export function validateLifestyle(l: AssessmentForm["lifestyle"]): Partial<Record<keyof AssessmentForm["lifestyle"], string>> {
  const e: Partial<Record<keyof AssessmentForm["lifestyle"], string>> = {};
  (Object.keys(l) as (keyof typeof l)[]).forEach((k) => {
    if (l[k] === null) e[k] = "Please choose an option.";
  });
  return e;
}

/** Absent symptoms submit 0 days. Throws if the form is invalid. */
export function buildPayload(form: AssessmentForm): AssessmentPayload {
  const ageErr = validateAge(form.age);
  if (ageErr) throw new Error(ageErr);
  const sexErr = validateSex(form.sex);
  if (sexErr) throw new Error(sexErr);
  const symptoms = {} as Record<SymptomKey, number>;
  for (const s of SYMPTOMS) {
    const a = form.symptoms[s.key];
    const err = validateSymptom(a);
    if (err) throw new Error(`${s.label}: ${err}`);
    symptoms[s.key] = a.present ? (a.days as number) : 0;
  }
  const l = form.lifestyle;
  if (l.smoking === null || l.alcohol === null || l.exercise === null) throw new Error("Lifestyle answers incomplete.");
  return {
    age: Number(form.age),
    sex: Number(form.sex),
    symptoms,
    lifestyle: { smoking: l.smoking, alcohol: l.alcohol, exercise: l.exercise },
  };
}
