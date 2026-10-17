import { useId, useState, type InputHTMLAttributes } from "react";
import { Eye, EyeOff } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
  hint?: string;
}

export function FormField({ label, error, hint, type, className, ...rest }: Props) {
  const id = useId();
  const [show, setShow] = useState(false);
  const isPw = type === "password";
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-sm font-medium text-ink">{label}</Label>
      <div className="relative">
        <Input
          id={id}
          type={isPw && show ? "text" : type}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-err` : hint ? `${id}-hint` : undefined}
          className={cn("h-11 bg-card", isPw && "pr-10", error && "border-destructive", className)}
          {...rest}
        />
        {isPw && (
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            className="absolute inset-y-0 right-0 grid w-10 place-items-center text-muted-foreground hover:text-ink focus-visible:outline-none focus-visible:text-ink"
            aria-label={show ? "Hide password" : "Show password"}
          >
            {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        )}
      </div>
      {error ? (
        <p id={`${id}-err`} className="text-xs text-destructive">{error}</p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-xs text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  );
}
