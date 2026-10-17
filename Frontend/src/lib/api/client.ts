/**
 * Centralized API client for the existing Flask backend.
 * - Base URL comes from VITE_API_BASE_URL (never hardcoded elsewhere).
 * - Attaches the access token; on 401 attempts ONE refresh, then retries once.
 * - Never logs tokens or passwords.
 */
import { tokenStore } from "./tokens";

export const API_BASE_URL = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:5000"
).replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

type Listener = () => void;
const sessionExpiredListeners = new Set<Listener>();
export function onSessionExpired(fn: Listener) {
  sessionExpiredListeners.add(fn);
  return () => sessionExpiredListeners.delete(fn);
}

function extractMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    for (const k of ["message", "error", "msg", "detail"]) {
      if (typeof b[k] === "string" && b[k]) return b[k] as string;
    }
    if (b.errors && typeof b.errors === "object") {
      const first = Object.values(b.errors as Record<string, unknown>)[0];
      if (typeof first === "string") return first;
      if (Array.isArray(first) && typeof first[0] === "string") return first[0];
    }
  }
  return fallback;
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { message: text.slice(0, 300) };
  }
}

let refreshPromise: Promise<boolean> | null = null;

/** POST /api/auth/refresh — refresh token sent as Bearer header (flask-jwt-extended default; UNVERIFIED). */
async function refreshAccessToken(): Promise<boolean> {
  const refresh = tokenStore.getRefresh();
  if (!refresh) return false;
  if (!refreshPromise) {
    refreshPromise = (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
          method: "POST",
          headers: { Authorization: `Bearer ${refresh}`, Accept: "application/json" },
        });
        if (!res.ok) return false;
        const body = (await parseBody(res)) as Record<string, unknown> | null;
        const token = body?.access_token;
        if (typeof token !== "string") return false;
        tokenStore.setAccess(token);
        if (typeof body?.refresh_token === "string") tokenStore.setRefresh(body.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        setTimeout(() => (refreshPromise = null), 0);
      }
    })();
  }
  return refreshPromise;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  auth?: boolean;
  query?: Record<string, string | number | undefined | null>;
  signal?: AbortSignal;
}

export async function apiRequest<T = unknown>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, auth = true, query, signal } = opts;
  const url = new URL(`${API_BASE_URL}${path}`);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
    }
  }

  const doFetch = async () => {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const token = tokenStore.getAccess();
    if (auth && token) headers.Authorization = `Bearer ${token}`;
    return fetch(url.toString(), {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });
  };

  let res: Response;
  try {
    res = await doFetch();
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(
      0,
      `Can't reach the server at ${API_BASE_URL}. Check that the backend is running and allows this origin (CORS).`,
      null,
    );
  }

  if (res.status === 401 && auth) {
    const ok = await refreshAccessToken();
    if (ok) {
      res = await doFetch();
    }
    if (res.status === 401) {
      tokenStore.clear();
      sessionExpiredListeners.forEach((fn) => fn());
      const b = await parseBody(res);
      throw new ApiError(401, "Your session has expired. Please sign in again.", b);
    }
  }

  const parsed = await parseBody(res);
  if (!res.ok) {
    const fallback =
      res.status === 403
        ? "You don't have permission to do that."
        : res.status === 404
          ? "We couldn't find that record."
          : res.status === 422 || res.status === 400
            ? "Some of the information wasn't accepted."
            : res.status >= 500
              ? "The server ran into a problem. Please try again shortly."
              : `Request failed (${res.status}).`;
    throw new ApiError(res.status, extractMessage(parsed, fallback), parsed);
  }
  if (parsed && typeof parsed === "object" && (parsed as Record<string, unknown>).success === false) {
    throw new ApiError(res.status, extractMessage(parsed, "The request was not successful."), parsed);
  }
  return parsed as T;
}
