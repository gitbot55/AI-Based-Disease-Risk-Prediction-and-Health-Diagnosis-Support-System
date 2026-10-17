/**
 * Typed service wrappers for KNOWN endpoints only.
 * Response types are intentionally loose (Record<string, unknown>) because the
 * exact shapes are UNVERIFIED against the Flask code. UI reads fields defensively.
 */
import { apiRequest } from "./client";
import type { AssessmentPayload } from "../assessment";

export type Json = Record<string, unknown>;

export const authApi = {
  login: (identifier: string, password: string) =>
    apiRequest<Json>("/api/auth/login", {
      method: "POST",
      auth: false,
      body: { identifier, password },
    }),
  register: (body: { username: string; email: string; password: string; full_name?: string }) =>
    apiRequest<Json>("/api/auth/register", { method: "POST", auth: false, body }),
  verify: () => apiRequest<Json>("/api/auth/verify"),
  logout: () => apiRequest<Json>("/api/auth/logout", { method: "DELETE" }),
};

export const predictionApi = {
  submit: (payload: AssessmentPayload) =>
    apiRequest<Json>("/api/prediction/submit", { method: "POST", body: payload }),
  get: (id: string) => apiRequest<Json>(`/api/prediction/${encodeURIComponent(id)}`),
  health: () => apiRequest<Json>("/api/prediction/health", { auth: false }),
};

export interface HistoryQuery {
  page?: number;
  per_page?: number;
  risk_level?: string;
  sort?: string;
}

export const dashboardApi = {
  summary: () => apiRequest<Json>("/api/dashboard/summary"),
  predictions: (q: HistoryQuery) =>
    apiRequest<Json>("/api/dashboard/predictions", { query: { ...q } }),
  prediction: (id: string) =>
    apiRequest<Json>(`/api/dashboard/predictions/${encodeURIComponent(id)}`),
  deletePrediction: (id: string) =>
    apiRequest<Json>(`/api/dashboard/predictions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  profile: () => apiRequest<Json>("/api/dashboard/profile"),
  updateProfile: (body: { full_name?: string; age?: number; gender?: string }) =>
    apiRequest<Json>("/api/dashboard/profile", { method: "PUT", body }),
  monthly: () => apiRequest<Json>("/api/dashboard/trends/monthly"),
  riskScore: () => apiRequest<Json>("/api/dashboard/trends/risk-score"),
};
