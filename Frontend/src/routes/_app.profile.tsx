import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, ErrorState, Loading, PageHeader } from "@/components/kit";
import { FormField } from "@/components/FormField";
import { Label } from "@/components/ui/label";
import { dashboardApi } from "@/lib/api/services";
import { useAuth } from "@/lib/auth";
import { isObj, num, str, unwrap } from "@/lib/read";

export const Route = createFileRoute("/_app/profile")({
  head: () => ({
    meta: [
      { title: "Profile — Glyco" },
      { name: "description", content: "Manage your name, age and gender." },
      { property: "og:title", content: "Profile — Glyco" },
      { property: "og:description", content: "Manage your name, age and gender." },
    ],
  }),
  component: Profile,
});

// Gender values are UNVERIFIED against the backend's accepted values.
const GENDERS = [{ v: "", l: "Prefer not to say" }, { v: "male", l: "Male" }, { v: "female", l: "Female" }, { v: "other", l: "Other" }];

function Profile() {
  const qc = useQueryClient();
  const { setUser } = useAuth();
  const q = useQuery({ queryKey: ["profile"], queryFn: dashboardApi.profile });
  const p = unwrap(q.data, ["profile", "user", "data"]);
  const [f, setF] = useState({ full_name: "", age: "", gender: "" });
  const [err, setErr] = useState<{ age?: string }>({});

  useEffect(() => {
    if (q.data) setF({ full_name: str(p, ["full_name"]) ?? "", age: num(p, ["age"])?.toString() ?? "", gender: str(p, ["gender"]) ?? "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q.data]);

  const save = useMutation({
    mutationFn: () => dashboardApi.updateProfile({
      full_name: f.full_name.trim() || undefined,
      age: f.age ? Number(f.age) : undefined,
      gender: f.gender || undefined,
    }),
    onSuccess: (res) => {
      toast.success("Profile saved.");
      const u = unwrap(res, ["profile", "user", "data"]);
      if (isObj(u) && Object.keys(u).length) setUser(u);
      qc.invalidateQueries({ queryKey: ["profile"] });
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't save."),
  });

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (f.age && (Number(f.age) < 1 || Number(f.age) > 120)) return setErr({ age: "Age must be between 1 and 120." });
    setErr({});
    save.mutate();
  }

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader eyebrow="Profile" title="Your details" />
      {q.isLoading ? <Loading /> : q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : (
        <Card>
          {(str(p, ["username"]) || str(p, ["email"])) && (
            <dl className="mb-6 grid gap-4 border-b border-border pb-6 text-sm sm:grid-cols-2">
              {str(p, ["username"]) && <div><dt className="text-xs text-muted-foreground">Username</dt><dd className="mt-0.5 font-medium">{str(p, ["username"])}</dd></div>}
              {str(p, ["email"]) && <div><dt className="text-xs text-muted-foreground">Email</dt><dd className="mt-0.5 font-medium">{str(p, ["email"])}</dd></div>}
            </dl>
          )}
          <form onSubmit={onSubmit} className="space-y-4" noValidate>
            <FormField label="Full name" value={f.full_name} onChange={(e) => setF({ ...f, full_name: e.target.value })} />
            <FormField label="Age" inputMode="numeric" value={f.age} onChange={(e) => setF({ ...f, age: e.target.value.replace(/\D/g, "").slice(0, 3) })} error={err.age} />
            <div className="space-y-1.5">
              <Label htmlFor="gender" className="text-sm font-medium text-ink">Gender</Label>
              <select id="gender" value={f.gender} onChange={(e) => setF({ ...f, gender: e.target.value })}
                className="h-11 w-full rounded-md border border-input bg-card px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                {GENDERS.map((g) => <option key={g.v} value={g.v}>{g.l}</option>)}
              </select>
            </div>
            <div className="pt-2"><Button type="submit" disabled={save.isPending} className="h-11">{save.isPending && <Loader2 className="size-4 animate-spin" />} Save changes</Button></div>
          </form>
        </Card>
      )}
    </div>
  );
}
