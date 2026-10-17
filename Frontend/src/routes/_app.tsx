import { createFileRoute, Link, Outlet, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { ClipboardPlus, History, LayoutGrid, LogOut, UserRound } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Loading } from "@/components/kit";
import { useAuth } from "@/lib/auth";
import { str } from "@/lib/read";

export const Route = createFileRoute("/_app")({
  component: AppLayout,
});

const NAV = [
  { to: "/dashboard", label: "Overview", icon: LayoutGrid },
  { to: "/assess", label: "New assessment", icon: ClipboardPlus },
  { to: "/history", label: "History", icon: History },
  { to: "/profile", label: "Profile", icon: UserRound },
] as const;

function AppLayout() {
  const { status, user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (status === "anonymous") navigate({ to: "/login", replace: true });
  }, [status, navigate]);

  if (status !== "authenticated") return <div className="grid min-h-screen place-items-center"><Loading label="Checking your session…" /></div>;

  const name = str(user ?? undefined, ["full_name", "username", "email"]);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_1fr]">
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-sidebar-border bg-sidebar px-4 py-6 lg:flex">
        <div className="px-2"><Logo /></div>
        <nav className="mt-10 flex flex-col gap-1" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-sidebar-foreground transition-colors hover:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
              activeProps={{ className: "bg-sidebar-accent text-sidebar-accent-foreground" }}
            >
              <Icon className="size-4" /> {label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto border-t border-sidebar-border pt-4">
          {name && <p className="truncate px-3 text-sm font-medium text-ink">{name}</p>}
          <button
            onClick={async () => { await logout(); navigate({ to: "/login" }); }}
            className="mt-2 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:bg-sidebar-accent hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          >
            <LogOut className="size-4" /> Sign out
          </button>
        </div>
      </aside>

      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-background/90 px-5 py-3 backdrop-blur lg:hidden">
        <Logo />
        <button
          onClick={async () => { await logout(); navigate({ to: "/login" }); }}
          className="rounded-md p-2 text-muted-foreground hover:text-ink"
          aria-label="Sign out"
        >
          <LogOut className="size-4" />
        </button>
      </header>

      <main className="px-5 pb-28 pt-8 sm:px-8 lg:px-12 lg:pb-16 lg:pt-12">
        <div className="mx-auto max-w-6xl fade-up"><Outlet /></div>
      </main>

      <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-4 border-t border-border bg-card/95 pb-[env(safe-area-inset-bottom)] backdrop-blur lg:hidden">
        {NAV.map(({ to, label, icon: Icon }) => (
          <Link
            key={to}
            to={to}
            className="flex flex-col items-center gap-1 py-2.5 text-[11px] font-medium text-muted-foreground"
            activeProps={{ className: "text-primary" }}
          >
            <Icon className="size-5" /> {label.replace("New assessment", "Assess")}
          </Link>
        ))}
      </nav>
    </div>
  );
}
