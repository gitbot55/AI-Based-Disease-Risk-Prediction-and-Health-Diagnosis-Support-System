import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Info, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Card, ErrorState, Loading, PageHeader } from "@/components/kit";
import { Recommendations, ResultSummary, SafetyNote, SymptomList } from "@/components/ResultView";
import { dashboardApi } from "@/lib/api/services";
import { persistedSymptoms, unwrap } from "@/lib/read";

export const Route = createFileRoute("/_app/history/$id")({
  head: () => ({
    meta: [
      { title: "Assessment details — Glyco" },
      { name: "description", content: "Details of a saved assessment." },
      { property: "og:title", content: "Assessment details — Glyco" },
      { property: "og:description", content: "Details of a saved assessment." },
    ],
  }),
  component: Detail,
});

function Detail() {
  const { id } = Route.useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["prediction", id], queryFn: () => dashboardApi.prediction(id) });
  const del = useMutation({
    mutationFn: () => dashboardApi.deletePrediction(id),
    onSuccess: () => { toast.success("Assessment deleted."); qc.invalidateQueries(); navigate({ to: "/history" }); },
    onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't delete."),
  });

  const data = unwrap(q.data);
  const symptoms = persistedSymptoms(data);

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Link to="/history" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-ink"><ArrowLeft className="size-4" /> All assessments</Link>
      <PageHeader eyebrow={`Record #${id}`} title="Assessment details"
        actions={q.data ? (
          <AlertDialog>
            <AlertDialogTrigger asChild><Button variant="outline"><Trash2 className="size-4" /> Delete</Button></AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader><AlertDialogTitle>Delete this assessment?</AlertDialogTitle><AlertDialogDescription>This permanently removes the record.</AlertDialogDescription></AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => del.mutate()} disabled={del.isPending} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">Delete</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        ) : undefined} />
      {q.isLoading ? <Loading /> : q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : (
        <>
          <ResultSummary data={data} />
          {symptoms ? <SymptomList symptoms={symptoms} /> : (
            <Card className="flex gap-3 text-sm text-muted-foreground">
              <Info className="mt-0.5 size-4 shrink-0" />
              <p>The symptom durations you entered aren't stored with past assessments, so they can't be shown here. Only the saved result is displayed.</p>
            </Card>
          )}
          <Recommendations data={data} />
          <SafetyNote />
        </>
      )}
    </div>
  );
}
