export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid size-8 place-items-center rounded-lg bg-ink text-primary-foreground">
        <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
          <path d="M12 3c3.5 4.2 6 7.6 6 10.7A6 6 0 0 1 6 13.7C6 10.6 8.5 7.2 12 3Z" />
          <path d="M9.5 14.5h5" />
        </svg>
      </div>
      <span className="display text-xl">Glyco</span>
    </div>
  );
}
