import { fetchAuthSession } from "aws-amplify/auth";
import { API_BASE_URL } from "./amplify-config";

async function getAuthToken(): Promise<string | null> {
  try {
    const session = await fetchAuthSession({ forceRefresh: false });
    const token = session.tokens?.idToken?.toString();
    return token || null;
  } catch {
    return null;
  }
}

export class ApiAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiAuthError";
  }
}

async function authedFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const token = await getAuthToken();
  if (!token) {
    throw new ApiAuthError("No valid auth token available");
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: token,
  };

  const res = await fetch(url, { ...init, headers });
  if (res.status === 401 || res.status === 403) {
    throw new ApiAuthError(`Auth rejected by API: ${res.status}`);
  }
  return res;
}

export async function apiGet<T = unknown>(path: string): Promise<T> {
  const res = await authedFetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`API GET ${path} failed: ${res.status}`);
  return res.json();
}

export async function apiPost<T = unknown>(
  path: string,
  body: unknown,
): Promise<T> {
  const res = await authedFetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API POST ${path} failed: ${res.status}`);
  return res.json();
}

export async function apiDelete<T = unknown>(path: string): Promise<T> {
  const res = await authedFetch(`${API_BASE_URL}${path}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`API DELETE ${path} failed: ${res.status}`);
  return res.json();
}

export async function apiPut<T = unknown>(
  path: string,
  body: unknown,
): Promise<T> {
  const res = await authedFetch(`${API_BASE_URL}${path}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API PUT ${path} failed: ${res.status}`);
  return res.json();
}
