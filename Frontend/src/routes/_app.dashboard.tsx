import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ClipboardPlus, LineChart as LineIcon } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Disclaimer, EmptyState, ErrorState, Loading, PageHeader, RiskBadge, Stat } from "@/components/kit";
import { dashboardApi } from "@/lib/api/services";
import { useAuth } from "@/lib/auth";
import { createdAt, formatDate, formatPct, isObj, listRows, num, probability, recordId, riskLevel, riskScore, riskTone, str, unwrap, type Obj } from "@/lib/read";

export const Route = createFileRoute("/_app/dashboard")({
  head: () => ({
    meta: [
      { title: "Overview — Glyco" },
      { name: "description", content: "Your assessment summary, trends and recent history." },
      { property: "og:title", content: "Overview — Glyco" },
      { property: "og:description", content: "Your assessment summary, trends and recent history." },
    ],
  }),
  component: Dashboard,
});

const toneVar = { low: "var(--risk-low)", mid: "var(--risk-mid)", high: "var(--risk-high)", neutral: "var(--chart-2)" };

/** Turn {"low": 3, "high": 1} or [{level, count}] into chart rows. Only real values. */
function toDistribution(v: unknown): { name: string; value: number }[] {
  if (Array.isArray(v)) {
    return v.filter(isObj).map((r) => ({ name: str(r, ["risk_level", "category", "level", "name", "label"]) ?? "?", value: num(r, ["count", "value", "total"]) ?? 0 }));
  }
  if (isObj(v)) return Object.entries(v).filter(([, n]) => typeof n === "number").map(([name, value]) => ({ name, value: value as number }));
  return [];
}

function Dashboard() {
  const { user } = useAuth();
  const summary = useQuery({ queryKey: ["summary"], queryFn: dashboardApi.summary });
  const recent = useQuery({ queryKey: ["predictions", { page: 1, per_page: 5 }], queryFn: () => dashboardApi.predictions({ page: 1, per_page: 5 }) });
  const monthly = useQuery({ queryKey: ["trend-monthly"], queryFn: dashboardApi.monthly });
  const score = useQuery({ queryKey: ["trend-score"], queryFn: dashboardApi.riskScore });

  const name = str(user ?? undefined, ["full_name", "username"]);
  const s = unwrap(summary.data, ["summary", "data"]);
  const total = num(s, ["total_predictions", "total", "count"]);
  const avg = num(s, ["average_risk_score", "avg_risk_score", "average_score"]);
  const dist = toDistribution(s.risk_distribution ?? s.distribution);
  const cats = toDistribution(s.category_breakdown ?? s.categories);
  const latest = isObj(s.latest_prediction) ? (s.latest_prediction as Obj) : isObj(s.latest) ? (s.latest as Obj) : undefined;
  const recentRows = listRows(recent.data);
  const monthlyRows = listRows(monthly.data);
  const scoreRows = listRows(score.data);

  return (
    <>
      <PageHeader
        eyebrow="Overview"
        title={name ? <>Hello, {name.split(" ")[0]}</> : "Your overview"}
        description="A summary of the assessments you've completed."
        actions={<Button asChild className="h-10"><Link to="/assess"><ClipboardPlus className="size-4" /> New assessment</Link></Button>}
      />

      {summary.isLoading ? (
        <Loading />
      ) : summary.error ? (
        <ErrorState error={summary.error} onRetry={() => summary.refetch()} />
      ) : total === 0 ? (
        <Card>
          <EmptyState
            icon={<ClipboardPlus className="size-5" />}
            title="No assessments yet"
            description="Start your first symptom assessment. It takes about two minutes."
            action={<Button asChild className="mt-2"><Link to="/assess">Start assessment</Link></Button>}
          />
        </Card>
      ) : (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Assessments" value={total ?? "—"} />
            <Stat label="Average risk score" value={avg !== undefined ? avg.toFixed(1) : "—"} hint={avg === undefined ? "Not provided by server" : undefined} />
            <Stat label="Latest result" value={latest ? <RiskBadge level={riskLevel(latest)} className="text-sm" /> : "—"} hint={latest ? formatDate(createdAt(latest)) : undefined} />
            <Stat label="Latest probability" value={latest && probability(latest) !== undefined ? formatPct(probability(latest)!) : "—"} />
          </div>

          <div className="grid gap-4 lg:grid-cols-5">
            <Card className="lg:col-span-3">
              <CardTitle>Risk score over time</CardTitle>
              <TrendChart q={score} rows={scoreRows} />
            </Card>
            <Card className="lg:col-span-2">
              <CardTitle>Risk distribution</CardTitle>
              {dist.length ? <DistChart data={dist} /> : <p className="py-10 text-center text-sm text-muted-foreground">Not provided by server.</p>}
            </Card>
          </div>

          <div className="grid gap-4 lg:grid-cols-5">
            <Card className="lg:col-span-3">
              <CardTitle aside={<Link to="/history" className="flex items-center gap-1 text-xs font-medium text-primary hover:underline">View all <ArrowRight className="size-3" /></Link>}>Recent assessments</CardTitle>
              {recent.isLoading ? <Loading /> : recent.error ? <ErrorState error={recent.error} onRetry={() => recent.refetch()} /> : recentRows.length === 0 ? (
                <p className="py-6 text-sm text-muted-foreground">No assessments returned.</p>
              ) : (
                <ul className="divide-y divide-border">
                  {recentRows.map((r, i) => {
                    const id = recordId(r);
                    const body = (
                      <div className="flex items-center justify-between gap-3 py-3">
                        <div>
                          <p className="text-sm font-medium text-ink">{formatDate(createdAt(r)) ?? (id ? `Record #${id}` : "Assessment")}</p>
                          {probability(r) !== undefined && <p className="text-xs text-muted-foreground tabular">Probability {formatPct(probability(r)!)}</p>}
                        </div>
                        <RiskBadge level={riskLevel(r)} />
                      </div>
                    );
                    return <li key={id ?? i}>{id ? <Link to="/history/$id" params={{ id }} className="block rounded-md px-1 hover:bg-muted/60">{body}</Link> : body}</li>;
                  })}
                </ul>
              )}
            </Card>
            <Card className="lg:col-span-2">
              <CardTitle>Monthly activity</CardTitle>
              <MonthlyChart q={monthly} rows={monthlyRows} />
              {cats.length > 0 && (
                <div className="mt-6 border-t border-border pt-4">
                  <p className="eyebrow mb-3">Category breakdown</p>
                  <ul className="space-y-2">
                    {cats.map((c) => (
                      <li key={c.name} className="flex justify-between text-sm"><span className="capitalize">{c.name.replace(/_/g, " ")}</span><span className="tabular font-medium">{c.value}</span></li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          </div>
          <Disclaimer compact />
        </div>
      )}
    </>
  );
}

type Q = { isLoading: boolean; error: unknown; refetch: () => unknown };

function TrendChart({ q, rows }: { q: Q; rows: Obj[] }) {
  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const data = rows
    .map((r) => ({ x: str(r, ["date", "created_at", "month", "label"]) ?? "", y: num(r, ["risk_score", "score", "average_risk_score", "value", "probability"]) }))
    .filter((d): d is { x: string; y: number } => d.y !== undefined);
  if (!data.length) return <EmptyState icon={<LineIcon className="size-5" />} title="No trend data yet" />;
  return (
    <div className="h-64">
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ left: -16, right: 8, top: 8 }}>
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.22} />
              <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis dataKey="x" tickLine={false} axisLine={false} fontSize={11} stroke="var(--muted-foreground)" tickFormatter={(v: string) => (v.length > 10 ? new Date(v).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : v)} />
          <YAxis tickLine={false} axisLine={false} fontSize={11} stroke="var(--muted-foreground)" />
          <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid var(--border)", fontSize: 12 }} />
          <Area type="monotone" dataKey="y" name="Score" stroke="var(--chart-1)" strokeWidth={2} fill="url(#g)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function MonthlyChart({ q, rows }: { q: Q; rows: Obj[] }) {
  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const data = rows
    .map((r) => ({ x: str(r, ["month", "label", "date"]) ?? "", y: num(r, ["count", "total", "predictions", "value"]) }))
    .filter((d): d is { x: string; y: number } => d.y !== undefined);
  if (!data.length) return <p className="py-10 text-center text-sm text-muted-foreground">No monthly data yet.</p>;
  return (
    <div className="h-44">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: -24, right: 4 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis dataKey="x" tickLine={false} axisLine={false} fontSize={11} stroke="var(--muted-foreground)" />
          <YAxis allowDecimals={false} tickLine={false} axisLine={false} fontSize={11} stroke="var(--muted-foreground)" />
          <Tooltip cursor={{ fill: "var(--muted)" }} contentStyle={{ borderRadius: 8, border: "1px solid var(--border)", fontSize: 12 }} />
          <Bar dataKey="y" name="Assessments" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function DistChart({ data }: { data: { name: string; value: number }[] }) {
  return (
    <div className="h-64">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" hide allowDecimals={false} />
          <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} fontSize={12} width={80} stroke="var(--muted-foreground)" />
          <Tooltip cursor={{ fill: "var(--muted)" }} contentStyle={{ borderRadius: 8, border: "1px solid var(--border)", fontSize: 12 }} />
          <Bar dataKey="value" name="Assessments" radius={[0, 4, 4, 0]} barSize={22}>
            {data.map((d) => <Cell key={d.name} fill={toneVar[riskTone(d.name)]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
