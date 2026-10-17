import { Card, CardTitle, Disclaimer, RiskBadge } from "@/components/kit";
import { SYMPTOMS, type SymptomKey } from "@/lib/assessment";
import { confidence, createdAt, formatDate, formatPct, probability, recommendations, recordId, riskLevel, riskScore, riskTone, shapExplanation, type Obj } from "@/lib/read";
import { cn } from "@/lib/utils";

const ring = { low: "text-risk-low", mid: "text-risk-mid", high: "text-risk-high", neutral: "text-chart-2" };

export function ResultSummary({ data }: { data: Obj }) {
  const level = riskLevel(data);
  const p = probability(data);
  const score = riskScore(data);
  const conf = confidence(data);
  const tone = riskTone(level);
  const pct = p !== undefined ? (p <= 1 ? p * 100 : p) : undefined;

  return (
    <Card className="p-6 sm:p-8">
      <div className="grid items-center gap-8 sm:grid-cols-[auto_1fr]">
        {pct !== undefined ? (
          <div className={cn("relative mx-auto size-40", ring[tone])}>
            <svg viewBox="0 0 100 100" className="size-full -rotate-90">
              <circle cx="50" cy="50" r="44" fill="none" stroke="var(--border)" strokeWidth="6" />
              <circle cx="50" cy="50" r="44" fill="none" stroke="currentColor" strokeWidth="6" strokeLinecap="round" strokeDasharray={`${(pct / 100) * 276.5} 276.5`} />
            </svg>
            <div className="absolute inset-0 grid place-items-center text-center">
              <div><p className="display tabular text-4xl text-ink">{formatPct(p!)}</p><p className="text-xs text-muted-foreground">estimated probability</p></div>
            </div>
          </div>
        ) : null}
        <div>
          <p className="eyebrow">Model estimate</p>
          <div className="mt-3"><RiskBadge level={level} className="px-3 py-1 text-sm" /></div>
          <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
            {score !== undefined && <Item k="Risk score" v={score.toFixed(score % 1 ? 2 : 0)} />}
            {conf !== undefined && <Item k="Confidence" v={formatPct(conf)} />}
            {formatDate(createdAt(data)) && <Item k="Date" v={formatDate(createdAt(data))!} />}
            {recordId(data) && <Item k="Record" v={`#${recordId(data)}`} />}
          </dl>
        </div>
      </div>
    </Card>
  );
}

function Item({ k, v }: { k: string; v: string }) {
  return <div><dt className="text-xs text-muted-foreground">{k}</dt><dd className="mt-0.5 font-medium tabular text-ink">{v}</dd></div>;
}

export function SymptomList({ symptoms, note }: { symptoms: Record<SymptomKey, number>; note?: string }) {
  return (
    <Card>
      <CardTitle>Symptoms you reported</CardTitle>
      {note && <p className="-mt-2 mb-3 text-xs text-muted-foreground">{note}</p>}
      <ul className="divide-y divide-border">
        {SYMPTOMS.map((s) => {
          const d = symptoms[s.key];
          return (
            <li key={s.key} className="flex items-center justify-between gap-4 py-2.5 text-sm">
              <span>{s.label}</span>
              <span className="flex items-center gap-3">
                {d > 0 && <span className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-muted sm:block"><span className="block h-full bg-primary" style={{ width: `${(d / 30) * 100}%` }} /></span>}
                <span className={cn("w-28 text-right tabular", d > 0 ? "font-medium text-ink" : "text-muted-foreground")}>{d > 0 ? `${d} ${d === 1 ? "day" : "days"}` : "Not experiencing"}</span>
              </span>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

export function Recommendations({ data }: { data: Obj }) {
  const recs = recommendations(data);
  if (!recs) return null;
  return (
    <Card>
      <CardTitle>Suggestions from the assessment</CardTitle>
      <ul className="space-y-2.5">
        {recs.map((r, i) => (
          <li key={i} className="flex gap-3 text-sm"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-primary" />{r}</li>
        ))}
      </ul>
    </Card>
  );
}

export function ShapFactors({ data }: { data: Obj }) {
  const shap = shapExplanation(data);
  if (!shap || !shap.feature_importance.length) return null;
  return (
    <Card>
      <CardTitle>Why did you get this score?</CardTitle>
      <p className="mb-4 text-sm text-muted-foreground">{shap.summary}</p>
      <div className="space-y-3">
        {shap.feature_importance.map((f, i) => (
          <div key={i} className="flex items-center justify-between text-sm">
            <span className="font-medium text-ink">{f.human_name}</span>
            <span className={cn("rounded-md px-2 py-1 text-xs font-semibold", f.direction === "positive" ? "bg-destructive/10 text-destructive" : f.direction === "negative" ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground")}>
              {f.direction === "positive" ? "+" : f.direction === "negative" ? "-" : ""}{Math.abs(f.shap_value).toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function SafetyNote() {
  return <Disclaimer />;
}
