import { describe, expect, it } from "vitest";
import { buildPayload, emptyForm, validateSymptom } from "./assessment";

function filled() {
  const f = emptyForm();
  f.age = "35";
  for (const k of Object.keys(f.symptoms) as (keyof typeof f.symptoms)[]) f.symptoms[k] = { present: false, days: null };
  f.lifestyle = { smoking: false, alcohol: false, exercise: true };
  return f;
}

describe("assessment rules", () => {
  it("absent symptoms submit 0 days", () => {
    const p = buildPayload(filled());
    expect(p.symptoms.fatigue_days).toBe(0);
    expect(p).toEqual({
      age: 35,
      symptoms: { frequent_urination_days: 0, excessive_thirst_days: 0, fatigue_days: 0, blurred_vision_days: 0, slow_healing_wounds_days: 0, weight_loss_days: 0 },
      lifestyle: { smoking: false, alcohol: false, exercise: true },
    });
  });
  it("present symptom submits its days", () => {
    const f = filled();
    f.symptoms.blurred_vision_days = { present: true, days: 7 };
    expect(buildPayload(f).symptoms.blurred_vision_days).toBe(7);
  });
  it("duration must be 1 to 30 days", () => {
    expect(validateSymptom({ present: true, days: 0 })).not.toBeNull();
    expect(validateSymptom({ present: true, days: 31 })).not.toBeNull();
    expect(validateSymptom({ present: true, days: 1 })).toBeNull();
    expect(validateSymptom({ present: true, days: 30 })).toBeNull();
  });
  it("absent symptom needs no duration", () => {
    expect(validateSymptom({ present: false, days: null })).toBeNull();
  });
});
