import { createFileRoute, Link } from "@tanstack/react-router";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ChevronLeft, ChevronRight, ClipboardPlus, History, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Card, EmptyState, ErrorState, Loading, PageHeader, RiskBadge } from "@/components/kit";
import { dashboardApi } from "@/lib/api/services";
import { createdAt, formatDate, formatPct, listRows, pagination, probability, recordId, riskLevel, riskScore } from "@/lib/read";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_app/history/")({
  head: () => ({
    meta: [
      { title: "Assessment history — Glyco" },
      { name: "description", content: "Browse, review and manage your past assessments." },
      { property: "og:title", content: "Assessment history — Glyco" },
      { property: "og:description", content: "Browse, review and manage your past assessments." },
    ],
  }),
  component: HistoryPage,
});

const PER_PAGE = 10;
// Filter values sent as `risk_level` query param — UNVERIFIED that the backend supports it.
const FILTERS = [{ v: "", l: "All" }, { v: "low", l: "Low" }, { v: "moderate", l: "Moderate" }, { v: "high", l: "High" }];

function HistoryPage() {
  const [page, setPage] = useState(1);
  const [risk, setRisk] = useState("");
  const [toDelete, setToDelete] = useState<string | null>(null);
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["predictions", { page, per_page: PER_PAGE, risk }],
    queryFn: () => dashboardApi.predictions({ page, per_page: PER_PAGE, risk_level: risk || undefined }),
    placeholderData: keepPreviousData,
  });
  const del = useMutation({
    mutationFn: (id: string) => dashboardApi.deletePrediction(id),
    onSuccess: () => { toast.success("Assessment deleted."); qc.invalidateQueries(); setToDelete(null); },
    onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't delete."),
  });

  const rows = listRows(q.data);
  const pg = pagination(q.data);
  const hasNext = pg.pages !== undefined ? page < pg.pages : rows.length === PER_PAGE;

  return (
    <>
      <PageHeader eyebrow="History" title="Your assessments" description={pg.total !== undefined ? `${pg.total} saved ${pg.total === 1 ? "assessment" : "assessments"}.` : undefined}
        actions={<Button asChild><Link to="/assess"><ClipboardPlus className="size-4" /> New</Link></Button>} />

      <div className="mb-4 flex flex-wrap gap-1.5" role="group" aria-label="Filter by risk">
        {FILTERS.map((f) => (
          <button key={f.v} onClick={() => { setRisk(f.v); setPage(1); }} aria-pressed={risk === f.v}
            className={cn("rounded-md border px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              risk === f.v ? "border-ink bg-ink text-primary-foreground" : "border-border bg-card text-muted-foreground hover:text-ink")}>
            {f.l}
          </button>
        ))}
      </div>

      <Card className="p-0 sm:p-0 overflow-hidden">
        {q.isLoading ? <Loading /> : q.error ? <div className="p-6"><ErrorState error={q.error} onRetry={() => q.refetch()} /></div> : rows.length === 0 ? (
          <EmptyState icon={<History className="size-5" />} title={risk ? "No assessments match this filter" : "No assessments yet"} description={risk ? "Try a different filter." : "Completed assessments will appear here."} />
        ) : (
          <ul className={cn("divide-y divide-border", q.isFetching && "opacity-60")}>
            {rows.map((r, i) => {
              const id = recordId(r);
              const p = probability(r);
              const s = riskScore(r);
              return (
                <li key={id ?? i} className="flex items-center gap-3 px-5 py-4 hover:bg-muted/40">
                  {id ? (
                    <Link to="/history/$id" params={{ id }} className="flex flex-1 flex-wrap items-center gap-x-6 gap-y-1 focus-visible:outline-none focus-visible:underline">
                      <span className="min-w-40 text-sm font-medium text-ink">{formatDate(createdAt(r)) ?? `Record #${id}`}</span>
                      <RiskBadge level={riskLevel(r)} />
                      {p !== undefined && <span className="text-sm text-muted-foreground tabular">Probability {formatPct(p)}</span>}
                      {s !== undefined && <span className="text-sm text-muted-foreground tabular">Score {s}</span>}
                    </Link>
                  ) : <span className="flex-1 text-sm">{formatDate(createdAt(r)) ?? "Assessment"}</span>}
                  {id && <Button variant="ghost" size="icon" aria-label="Delete assessment" onClick={() => setToDelete(id)}><Trash2 className="size-4 text-muted-foreground" /></Button>}
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      {(page > 1 || hasNext) && (
        <div className="mt-4 flex items-center justify-between text-sm">
          <span className="text-muted-foreground tabular">Page {page}{pg.pages ? ` of ${pg.pages}` : ""}</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}><ChevronLeft className="size-4" /> Previous</Button>
            <Button variant="outline" size="sm" disabled={!hasNext} onClick={() => setPage((p) => p + 1)}>Next <ChevronRight className="size-4" /></Button>
          </div>
        </div>
      )}

      <AlertDialog open={!!toDelete} onOpenChange={(o) => !o && setToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this assessment?</AlertDialogTitle>
            <AlertDialogDescription>This permanently removes the record. It can't be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={del.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction disabled={del.isPending} onClick={(e) => { e.preventDefault(); if (toDelete) del.mutate(toDelete); }} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              {del.isPending ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
