import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { AuthLayout } from "@/components/AuthLayout";
import { FormField } from "@/components/FormField";
import { Button } from "@/components/ui/button";
import { authApi } from "@/lib/api/services";
import { ApiError } from "@/lib/api/client";
import { isObj } from "@/lib/read";

export const Route = createFileRoute("/register")({
  head: () => ({
    meta: [
      { title: "Create account — Glyco" },
      { name: "description", content: "Create an account to save your symptom assessments." },
      { property: "og:title", content: "Create account — Glyco" },
      { property: "og:description", content: "Create an account to save your symptom assessments." },
    ],
  }),
  component: RegisterPage,
});

type F = { full_name: string; username: string; email: string; password: string; confirm: string };

function RegisterPage() {
  const navigate = useNavigate();
  const [f, setF] = useState<F>({ full_name: "", username: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState<Partial<Record<keyof F | "form", string>>>({});
  const [busy, setBusy] = useState(false);
  const set = (k: keyof F) => (e: React.ChangeEvent<HTMLInputElement>) => setF({ ...f, [k]: e.target.value });

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const errs: typeof errors = {};
    if (!f.username.trim()) errs.username = "Choose a username.";
    if (!/^\S+@\S+\.\S+$/.test(f.email)) errs.email = "Enter a valid email address.";
    if (f.password.length < 8) errs.password = "Use at least 8 characters.";
    if (f.confirm !== f.password) errs.confirm = "Passwords don't match.";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    setBusy(true);
    try {
      await authApi.register({
        username: f.username.trim(),
        email: f.email.trim(),
        password: f.password,
        ...(f.full_name.trim() ? { full_name: f.full_name.trim() } : {}),
      });
      toast.success("Account created. Please sign in.");
      navigate({ to: "/login" });
    } catch (err) {
      const fieldErrs: typeof errors = {};
      if (err instanceof ApiError && isObj(err.body) && isObj(err.body.errors)) {
        for (const [k, v] of Object.entries(err.body.errors)) {
          const m = Array.isArray(v) ? v[0] : v;
          if (typeof m === "string" && k in f) fieldErrs[k as keyof F] = m;
        }
      }
      setErrors({ ...fieldErrs, form: err instanceof Error ? err.message : "Registration failed." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Save assessments and review them over time."
      footer={<>Already have an account? <Link to="/login" className="font-medium text-primary hover:underline">Sign in</Link></>}
    >
      <form onSubmit={onSubmit} className="space-y-4" noValidate>
        <FormField label="Full name (optional)" autoComplete="name" value={f.full_name} onChange={set("full_name")} error={errors.full_name} />
        <FormField label="Username" autoComplete="username" value={f.username} onChange={set("username")} error={errors.username} />
        <FormField label="Email" type="email" autoComplete="email" value={f.email} onChange={set("email")} error={errors.email} />
        <FormField label="Password" type="password" autoComplete="new-password" value={f.password} onChange={set("password")} error={errors.password} hint="At least 8 characters." />
        <FormField label="Confirm password" type="password" autoComplete="new-password" value={f.confirm} onChange={set("confirm")} error={errors.confirm} />
        {errors.form && <p role="alert" className="rounded-md bg-destructive/5 px-3 py-2 text-sm text-destructive">{errors.form}</p>}
        <Button type="submit" className="h-11 w-full" disabled={busy}>
          {busy && <Loader2 className="size-4 animate-spin" />} Create account
        </Button>
      </form>
    </AuthLayout>
  );
}
