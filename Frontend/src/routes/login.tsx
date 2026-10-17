import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState, type FormEvent } from "react";
import { Loader2 } from "lucide-react";
import { AuthLayout } from "@/components/AuthLayout";
import { FormField } from "@/components/FormField";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — Glyco" },
      { name: "description", content: "Sign in to view your symptom assessments and history." },
      { property: "og:title", content: "Sign in — Glyco" },
      { property: "og:description", content: "Sign in to view your symptom assessments and history." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const { login, status, expired } = useAuth();
  const navigate = useNavigate();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<{ identifier?: string; password?: string; form?: string }>({});
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (status === "authenticated") navigate({ to: "/dashboard" });
  }, [status, navigate]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const errs: typeof errors = {};
    if (!identifier.trim()) errs.identifier = "Enter your email or username.";
    if (!password) errs.password = "Enter your password.";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    setBusy(true);
    try {
      await login(identifier.trim(), password);
      navigate({ to: "/dashboard" });
    } catch (err) {
      setErrors({ form: err instanceof Error ? err.message : "Sign-in failed." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Sign in with your email or username."
      footer={<>New here? <Link to="/register" className="font-medium text-primary hover:underline">Create an account</Link></>}
    >
      {expired && !errors.form && (
        <p className="mb-5 rounded-md bg-risk-mid-soft px-3 py-2 text-sm text-ink">Your session expired. Please sign in again.</p>
      )}
      <form onSubmit={onSubmit} className="space-y-4" noValidate>
        <FormField label="Email or username" autoComplete="username" value={identifier} onChange={(e) => setIdentifier(e.target.value)} error={errors.identifier} />
        <FormField label="Password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} error={errors.password} />
        {errors.form && <p role="alert" className="rounded-md bg-destructive/5 px-3 py-2 text-sm text-destructive">{errors.form}</p>}
        <Button type="submit" className="h-11 w-full" disabled={busy}>
          {busy && <Loader2 className="size-4 animate-spin" />} Sign in
        </Button>
      </form>
    </AuthLayout>
  );
}
