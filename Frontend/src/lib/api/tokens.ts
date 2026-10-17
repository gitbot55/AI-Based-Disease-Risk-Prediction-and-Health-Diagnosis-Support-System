/**
 * Token storage. Uses localStorage so sessions survive reloads.
 * SECURITY LIMITATION: localStorage is readable by any script on the origin (XSS risk).
 * httpOnly cookies would require backend changes. See INTEGRATION.md.
 */
const ACCESS = "dh.access";
const REFRESH = "dh.refresh";
const USER = "dh.user";

const safe = () => (typeof window === "undefined" ? null : window.localStorage);

export const tokenStore = {
  getAccess: () => safe()?.getItem(ACCESS) ?? null,
  getRefresh: () => safe()?.getItem(REFRESH) ?? null,
  setAccess: (t: string) => safe()?.setItem(ACCESS, t),
  setRefresh: (t: string) => safe()?.setItem(REFRESH, t),
  getUser: (): Record<string, unknown> | null => {
    const raw = safe()?.getItem(USER);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  },
  setUser: (u: unknown) => safe()?.setItem(USER, JSON.stringify(u ?? null)),
  clear: () => {
    const s = safe();
    s?.removeItem(ACCESS);
    s?.removeItem(REFRESH);
    s?.removeItem(USER);
  },
};
