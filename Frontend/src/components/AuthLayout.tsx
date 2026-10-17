import type { ReactNode } from "react";
import { Logo } from "./Logo";

export function AuthLayout({ title, subtitle, children, footer }: { title: string; subtitle: string; children: ReactNode; footer: ReactNode }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-[1fr_1.1fr]">
      <aside className="relative hidden flex-col justify-between border-r border-border bg-surface p-12 lg:flex">
        <Logo />
        <div className="max-w-md">
          <p className="eyebrow mb-4">Symptom-first assessment</p>
          <h2 className="display text-4xl leading-tight">
            Understand what your symptoms may be telling you — calmly, clearly.
          </h2>
          <p className="mt-5 text-muted-foreground">
            Answer a few questions about how you've been feeling and for how many days. You'll receive a model-based estimate — never a diagnosis.
          </p>
        </div>
        <p className="text-xs text-muted-foreground">For information only. Always consult a qualified healthcare professional.</p>
      </aside>
      <main className="flex items-center justify-center px-5 py-12">
        <div className="w-full max-w-sm fade-up">
          <div className="mb-10 lg:hidden"><Logo /></div>
          <h1 className="display text-3xl">{title}</h1>
          <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>
          <div className="mt-8">{children}</div>
          <div className="mt-8 text-sm text-muted-foreground">{footer}</div>
        </div>
      </main>
    </div>
  );
}
