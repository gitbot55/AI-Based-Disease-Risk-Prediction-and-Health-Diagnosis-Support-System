import type { ReactNode } from "react";
import { AlertCircle, Info, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { riskTone, type RiskTone } from "@/lib/read";
import { ApiError } from "@/lib/api/client";

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
        <h1 className="display text-3xl sm:text-4xl">{title}</h1>
        {description && <p className="mt-2 text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </div>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("surface-card p-5 sm:p-6", className)}>{children}</div>;
}

export function CardTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-2">
      <h2 className="text-sm font-semibold text-ink">{children}</h2>
      {aside}
    </div>
  );
}

const toneClass: Record<RiskTone, string> = {
  low: "bg-risk-low-soft text-risk-low",
  mid: "bg-risk-mid-soft text-risk-mid",
  high: "bg-risk-high-soft text-risk-high",
  neutral: "bg-muted text-muted-foreground",
};

export function RiskBadge({ level, className }: { level?: string; className?: string }) {
  if (!level) return <span className="text-sm text-muted-foreground">—</span>;
  const tone = riskTone(level);
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-semibold capitalize", toneClass[tone], className)}>
      <span className="size-1.5 rounded-full bg-current" aria-hidden />
      {level.replace(/_/g, " ")}
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-10 text-sm text-muted-foreground justify-center">
      <Loader2 className="size-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const status = error instanceof ApiError ? error.status : undefined;
  const msg = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div role="alert" className="flex flex-col items-center gap-3 rounded-lg border border-destructive/20 bg-destructive/5 px-6 py-8 text-center">
      <AlertCircle className="size-5 text-destructive" />
      <div>
        <p className="font-medium text-ink">{status === 404 ? "Not found" : status === 403 ? "Access denied" : "Couldn't load this"}</p>
        <p className="mt-1 text-sm text-muted-foreground">{msg}</p>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="size-3.5" /> Try again
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ icon, title, description, action }: { icon?: ReactNode; title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 px-6 py-10 text-center">
      {icon && <div className="grid size-10 place-items-center rounded-lg bg-accent text-accent-foreground">{icon}</div>}
      <p className="font-medium text-ink">{title}</p>
      {description && <p className="max-w-sm text-sm text-muted-foreground">{description}</p>}
      {action}
    </div>
  );
}

export function Disclaimer({ compact }: { compact?: boolean }) {
  return (
    <div className="flex gap-3 rounded-lg border border-border bg-surface p-4 text-sm text-muted-foreground">
      <Info className="mt-0.5 size-4 shrink-0 text-primary" />
      <p>
        {compact
          ? "This is a model-based estimate for information only — not a diagnosis."
          : "This is a model-based estimate for information only. It cannot confirm or rule out diabetes, and symptoms can have many causes. If you're concerned, please speak with a qualified healthcare professional — clinical testing may be needed."}
      </p>
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <Card className="p-5">
      <p className="eyebrow">{label}</p>
      <p className="display tabular mt-3 text-3xl">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </Card>
  );
}
