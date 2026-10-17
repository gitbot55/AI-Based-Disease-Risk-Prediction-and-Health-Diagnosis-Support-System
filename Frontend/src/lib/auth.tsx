import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { onSessionExpired } from "./api/client";
import { authApi, type Json } from "./api/services";
import { tokenStore } from "./api/tokens";
import { isObj } from "./read";
import type { AssessmentPayload } from "./assessment";

type Status = "loading" | "authenticated" | "anonymous";

interface LastResult {
  payload: AssessmentPayload;
  response: Json;
  submittedAt: string;
}

interface AuthCtx {
  status: Status;
  user: Json | null;
  expired: boolean;
  login: (identifier: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  setUser: (u: Json | null) => void;
  lastResult: LastResult | null;
  setLastResult: (r: LastResult | null) => void;
}

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>("loading");
  const [user, setUserState] = useState<Json | null>(null);
  const [expired, setExpired] = useState(false);
  const [lastResult, setLastResult] = useState<LastResult | null>(null);

  const setUser = useCallback((u: Json | null) => {
    setUserState(u);
    if (u) tokenStore.setUser(u);
  }, []);

  useEffect(() => {
    const off = onSessionExpired(() => {
      setExpired(true);
      setUserState(null);
      setStatus("anonymous");
    });
    if (!tokenStore.getAccess()) {
      setStatus("anonymous");
      return () => void off();
    }
    setUserState(tokenStore.getUser() as Json | null);
    authApi
      .verify()
      .then((res) => {
        const u = isObj(res.user) ? (res.user as Json) : null;
        if (u) setUser(u);
        setStatus("authenticated");
      })
      .catch(() => {
        tokenStore.clear();
        setStatus("anonymous");
      });
    return () => void off();
  }, [setUser]);

  const login = useCallback(
    async (identifier: string, password: string) => {
      const res = await authApi.login(identifier, password);
      const access = res.access_token;
      if (typeof access !== "string") throw new Error("The server did not return an access token.");
      tokenStore.setAccess(access);
      if (typeof res.refresh_token === "string") tokenStore.setRefresh(res.refresh_token);
      setUser(isObj(res.user) ? (res.user as Json) : null);
      setExpired(false);
      setStatus("authenticated");
    },
    [setUser],
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      /* token may already be invalid; clear locally regardless */
    }
    tokenStore.clear();
    setUserState(null);
    setLastResult(null);
    setStatus("anonymous");
  }, []);

  const value = useMemo(
    () => ({ status, user, expired, login, logout, setUser, lastResult, setLastResult }),
    [status, user, expired, login, logout, setUser, lastResult],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAuth must be used within AuthProvider");
  return c;
}
