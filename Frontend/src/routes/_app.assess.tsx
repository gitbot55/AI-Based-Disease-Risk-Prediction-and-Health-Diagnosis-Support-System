import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, Loader2, Minus, Pencil, Plus } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, Disclaimer, PageHeader } from "@/components/kit";
import { FormField } from "@/components/FormField";
import { cn } from "@/lib/utils";
import {
  MAX_DAYS, MIN_DAYS, SYMPTOMS, buildPayload, emptyForm, validateAge, validateLifestyle, validateSymptom, validateSex,
  type AssessmentForm, type SymptomKey,
} from "@/lib/assessment";
import { predictionApi } from "@/lib/api/services";
import { useAuth } from "@/lib/auth";

export const Route = createFileRoute("/_app/assess")({
  head: () => ({
    meta: [
      { title: "New assessment — Glyco" },
      { name: "description", content: "Answer questions about your symptoms and how many days you've had them." },
      { property: "og:title", content: "New assessment — Glyco" },
      { property: "og:description", content: "Answer questions about your symptoms and how many days you've had them." },
    ],
  }),
  component: Assess,
});

const STEPS = ["About you", "Symptoms", "Lifestyle", "Review"] as const;
const LIFESTYLE = [
  { key: "smoking", label: "Do you currently smoke?", hint: "Any regular tobacco smoking." },
  { key: "alcohol", label: "Do you drink alcohol regularly?" },
  { key: "exercise", label: "Do you exercise regularly?", hint: "For example, brisk activity most weeks." },
] as const;

function Assess() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<AssessmentForm>(emptyForm);
  const [showErrors, setShowErrors] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const inflight = useRef(false);
  const { setLastResult } = useAuth();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const top = useRef<HTMLDivElement>(null);

  const ageErr = validateAge(form.age);
  const sexErr = validateSex(form.sex);
  const symErrs = Object.fromEntries(SYMPTOMS.map((s) => [s.key, validateSymptom(form.symptoms[s.key])])) as Record<SymptomKey, string | null>;
  const lifeErrs = validateLifestyle(form.lifestyle);
  const stepValid = [!ageErr && !sexErr, Object.values(symErrs).every((e) => !e), Object.keys(lifeErrs).length === 0, true];

  const go = (n: number) => { setStep(n); setShowErrors(false); top.current?.scrollIntoView({ behavior: "smooth", block: "start" }); };
  const next = () => (stepValid[step] ? go(step + 1) : setShowErrors(true));

  const setSym = (k: SymptomKey, patch: Partial<AssessmentForm["symptoms"][SymptomKey]>) =>
    setForm((f) => ({ ...f, symptoms: { ...f.symptoms, [k]: { ...f.symptoms[k], ...patch } } }));

  async function submit() {
    if (inflight.current) return;
    setSubmitError(null);
    let payload;
    try { payload = buildPayload(form); } catch (e) { setSubmitError((e as Error).message); return; }
    inflight.current = true;
    setSubmitting(true);
    try {
      const response = await predictionApi.submit(payload);
      setLastResult({ payload, response, submittedAt: new Date().toISOString() });
      qc.invalidateQueries();
      navigate({ to: "/results" });
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "Submission failed.");
    } finally {
      inflight.current = false;
      setSubmitting(false);
    }
  }

  return (
    <div ref={top} className="mx-auto max-w-3xl scroll-mt-24">
      <PageHeader eyebrow="New assessment" title="How have you been feeling?" description="Please report your own experiences as accurately as you can. There are no right or wrong answers." />

      <ol className="mb-8 grid grid-cols-4 gap-2" aria-label="Progress">
        {STEPS.map((label, i) => (
          <li key={label}>
            <button
              type="button"
              disabled={i > step}
              onClick={() => i < step && go(i)}
              aria-current={i === step ? "step" : undefined}
              className="group w-full text-left disabled:cursor-default"
            >
              <span className={cn("block h-1 rounded-full transition-colors", i <= step ? "bg-primary" : "bg-border")} />
              <span className={cn("mt-2 block text-xs font-medium", i === step ? "text-ink" : "text-muted-foreground", i < step && "group-hover:text-primary")}>
                <span className="tabular">{i + 1}.</span> {label}
              </span>
            </button>
          </li>
        ))}
      </ol>

      {step === 0 && (
        <Card>
          <h2 className="text-lg font-semibold">About you</h2>
          <p className="mt-1 text-sm text-muted-foreground">Your age and sex help the model put your answers into context.</p>
          <div className="mt-6 max-w-xs space-y-6">
            <FormField label="Age" inputMode="numeric" value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value.replace(/[^\d]/g, "").slice(0, 3) })} error={showErrors ? ageErr ?? undefined : undefined} hint="In years." />
            
            <fieldset>
              <legend className="mb-2 text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Biological Sex</legend>
              <div className="grid grid-cols-2 gap-2" role="radiogroup">
                <Choice selected={form.sex === "1"} onClick={() => setForm({ ...form, sex: "1" })}>Male</Choice>
                <Choice selected={form.sex === "0"} onClick={() => setForm({ ...form, sex: "0" })}>Female</Choice>
              </div>
              {showErrors && form.sex === "" && <p className="mt-2 text-[0.8rem] font-medium text-destructive">Please select your biological sex.</p>}
            </fieldset>
          </div>
        </Card>
      )}

      {step === 1 && (
        <div className="space-y-3">
          {SYMPTOMS.map((s) => {
            const a = form.symptoms[s.key];
            const err = showErrors ? symErrs[s.key] : null;
            return (
              <Card key={s.key} className={cn("p-5", err && "border-destructive/50")}>
                <fieldset>
                  <legend className="font-semibold text-ink">{s.label}</legend>
                  <p className="mt-0.5 text-sm text-muted-foreground">{s.hint}</p>
                  <div className="mt-4 grid gap-2 sm:grid-cols-2" role="radiogroup">
                    <Choice selected={a.present === true} onClick={() => setSym(s.key, { present: true, days: a.days ?? 1 })}>I've experienced this</Choice>
                    <Choice selected={a.present === false} onClick={() => setSym(s.key, { present: false, days: null })}>Not experiencing this symptom</Choice>
                  </div>
                  {a.present && <DaysInput label={s.label} value={a.days} onChange={(d) => setSym(s.key, { days: d })} />}
                  {err && <p className="mt-3 text-xs text-destructive">{err}</p>}
                </fieldset>
              </Card>
            );
          })}
        </div>
      )}

      {step === 2 && (
        <div className="space-y-3">
          {LIFESTYLE.map((q) => {
            const v = form.lifestyle[q.key];
            const err = showErrors ? lifeErrs[q.key] : undefined;
            return (
              <Card key={q.key} className={cn("p-5", err && "border-destructive/50")}>
                <fieldset>
                  <legend className="font-semibold text-ink">{q.label}</legend>
                  {"hint" in q && <p className="mt-0.5 text-sm text-muted-foreground">{q.hint}</p>}
                  <div className="mt-4 grid grid-cols-2 gap-2" role="radiogroup">
                    <Choice selected={v === true} onClick={() => setForm({ ...form, lifestyle: { ...form.lifestyle, [q.key]: true } })}>Yes</Choice>
                    <Choice selected={v === false} onClick={() => setForm({ ...form, lifestyle: { ...form.lifestyle, [q.key]: false } })}>No</Choice>
                  </div>
                  {err && <p className="mt-3 text-xs text-destructive">{err}</p>}
                </fieldset>
              </Card>
            );
          })}
        </div>
      )}

      {step === 3 && (
        <div className="space-y-4">
          <Card>
            <ReviewHeader title="About you" onEdit={() => go(0)} />
            <p className="text-sm">Age <span className="ml-2 font-semibold tabular">{form.age}</span></p>
            <p className="text-sm mt-1">Sex <span className="ml-2 font-semibold tabular">{form.sex === "1" ? "Male" : "Female"}</span></p>
          </Card>
          <Card>
            <ReviewHeader title="Symptoms and duration" onEdit={() => go(1)} />
            <dl className="divide-y divide-border">
              {SYMPTOMS.map((s) => {
                const a = form.symptoms[s.key];
                return (
                  <div key={s.key} className="flex justify-between py-2.5 text-sm">
                    <dt>{s.label}</dt>
                    <dd className={cn("tabular font-medium", !a.present && "text-muted-foreground font-normal")}>{a.present ? `${a.days} ${a.days === 1 ? "day" : "days"}` : "Not experiencing"}</dd>
                  </div>
                );
              })}
            </dl>
          </Card>
          <Card>
            <ReviewHeader title="Lifestyle" onEdit={() => go(2)} />
            <dl className="divide-y divide-border">
              {LIFESTYLE.map((q) => (
                <div key={q.key} className="flex justify-between py-2.5 text-sm"><dt>{q.label}</dt><dd className="font-medium">{form.lifestyle[q.key] ? "Yes" : "No"}</dd></div>
              ))}
            </dl>
          </Card>
          <Disclaimer />
          {submitError && <p role="alert" className="rounded-md border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">{submitError} Your answers are still here — you can try again.</p>}
        </div>
      )}

      <div className="mt-8 flex items-center justify-between">
        <Button variant="ghost" onClick={() => go(step - 1)} disabled={step === 0 || submitting}><ArrowLeft className="size-4" /> Back</Button>
        {step < 3 ? (
          <Button onClick={next} className="h-11 px-6">Continue <ArrowRight className="size-4" /></Button>
        ) : (
          <Button onClick={submit} disabled={submitting} className="h-11 px-6">
            {submitting ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />} {submitting ? "Analysing…" : "Submit assessment"}
          </Button>
        )}
      </div>
    </div>
  );
}

function Choice({ selected, onClick, children }: { selected: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 rounded-lg border px-4 py-3 text-left text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected ? "border-primary bg-accent text-accent-foreground" : "border-input bg-card hover:border-ring/60",
      )}
    >
      <span className={cn("grid size-4 shrink-0 place-items-center rounded-full border", selected ? "border-primary bg-primary" : "border-input")}>
        {selected && <span className="size-1.5 rounded-full bg-primary-foreground" />}
      </span>
      {children}
    </button>
  );
}

function DaysInput({ label, value, onChange }: { label: string; value: number | null; onChange: (n: number | null) => void }) {
  const v = value ?? MIN_DAYS;
  const clamp = (n: number) => Math.min(MAX_DAYS, Math.max(MIN_DAYS, n));
  return (
    <div className="mt-5 rounded-lg bg-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <label className="text-sm text-muted-foreground" htmlFor={`days-${label}`}>For how many days?</label>
        <div className="flex items-center gap-2">
          <Button type="button" variant="outline" size="icon" className="size-9" onClick={() => onChange(clamp(v - 1))} disabled={v <= MIN_DAYS} aria-label="One day fewer"><Minus className="size-4" /></Button>
          <input
            id={`days-${label}`}
            type="number"
            inputMode="numeric"
            min={MIN_DAYS}
            max={MAX_DAYS}
            value={value ?? ""}
            onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
            className="h-9 w-16 rounded-md border border-input bg-card text-center font-semibold tabular focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <Button type="button" variant="outline" size="icon" className="size-9" onClick={() => onChange(clamp(v + 1))} disabled={v >= MAX_DAYS} aria-label="One day more"><Plus className="size-4" /></Button>
          <span className="w-12 text-sm font-medium text-ink">{v === 1 ? "day" : "days"}</span>
        </div>
      </div>
      <input
        type="range"
        min={MIN_DAYS}
        max={MAX_DAYS}
        value={v}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={`${label} duration in days`}
        aria-valuetext={`${v} days`}
        className="mt-4 w-full accent-primary"
      />
      <div className="flex justify-between text-[11px] text-muted-foreground tabular"><span>1 day</span><span>30 days</span></div>
    </div>
  );
}

function ReviewHeader({ title, onEdit }: { title: string; onEdit: () => void }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <h3 className="text-sm font-semibold">{title}</h3>
      <Button variant="ghost" size="sm" onClick={onEdit}><Pencil className="size-3.5" /> Edit</Button>
    </div>
  );
}
