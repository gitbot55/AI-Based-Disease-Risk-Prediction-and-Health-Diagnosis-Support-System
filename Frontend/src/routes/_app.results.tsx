import { createFileRoute, Link } from "@tanstack/react-router";
import { ClipboardPlus, FileText, LayoutGrid } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, EmptyState, PageHeader } from "@/components/kit";
import { Recommendations, ResultSummary, SafetyNote, SymptomList, ShapFactors } from "@/components/ResultView";
import { useAuth } from "@/lib/auth";
import { recordId, unwrap } from "@/lib/read";

export const Route = createFileRoute("/_app/results")({
  head: () => ({
    meta: [
      { title: "Your result — Glyco" },
      { name: "description", content: "The model-based estimate from your latest symptom assessment." },
      { property: "og:title", content: "Your result — Glyco" },
      { property: "og:description", content: "The model-based estimate from your latest symptom assessment." },
    ],
  }),
  component: Results,
});

function Results() {
  const { lastResult } = useAuth();
  if (!lastResult) {
    return (
      <Card>
        <EmptyState icon={<FileText className="size-5" />} title="No result to show" description="Results appear here right after you submit an assessment. Past results are in your history."
          action={<div className="mt-2 flex gap-2"><Button asChild><Link to="/assess">New assessment</Link></Button><Button asChild variant="outline"><Link to="/history">History</Link></Button></div>} />
      </Card>
    );
  }
  const data = unwrap(lastResult.response);
  const id = recordId(data);
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <PageHeader eyebrow="Assessment complete" title="Your result" description="Here's what the model estimated from the answers you just gave." />
      <ResultSummary data={{ created_at: lastResult.submittedAt, ...data }} />
      <ShapFactors data={data} />
      <SymptomList symptoms={lastResult.payload.symptoms} note={`Age ${lastResult.payload.age} · Smoking ${lastResult.payload.lifestyle.smoking ? "yes" : "no"} · Alcohol ${lastResult.payload.lifestyle.alcohol ? "yes" : "no"} · Exercise ${lastResult.payload.lifestyle.exercise ? "yes" : "no"}`} />
      <Recommendations data={data} />
      <SafetyNote />
      <div className="flex flex-wrap gap-2 pt-2">
        <Button asChild><Link to="/dashboard"><LayoutGrid className="size-4" /> Back to overview</Link></Button>
        {id && <Button asChild variant="outline"><Link to="/history/$id" params={{ id }}><FileText className="size-4" /> View saved record</Link></Button>}
        <Button asChild variant="ghost"><Link to="/assess"><ClipboardPlus className="size-4" /> New assessment</Link></Button>
      </div>
    </div>
  );
}
