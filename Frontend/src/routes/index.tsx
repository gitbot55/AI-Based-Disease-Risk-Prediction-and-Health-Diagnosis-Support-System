import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { Loading } from "@/components/kit";
import { useAuth } from "@/lib/auth";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Glyco — Symptom-based diabetes risk insights" },
      { name: "description", content: "Answer symptom questions and receive a calm, model-based diabetes risk estimate." },
      { property: "og:title", content: "Glyco — Symptom-based diabetes risk insights" },
      { property: "og:description", content: "Answer symptom questions and receive a calm, model-based diabetes risk estimate." },
    ],
  }),
  component: Index,
});

function Index() {
  const { status } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    if (status === "authenticated") navigate({ to: "/dashboard", replace: true });
    if (status === "anonymous") navigate({ to: "/login", replace: true });
  }, [status, navigate]);
  return <div className="min-h-screen grid place-items-center"><Loading /></div>;
}
