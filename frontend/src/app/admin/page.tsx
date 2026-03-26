"use client";

import "@/lib/amplify-config";
import { useEffect, useState } from "react";
import { getCurrentUser, signInWithRedirect } from "aws-amplify/auth";
import { apiGet, apiPost, apiDelete, ApiAuthError } from "@/lib/api";
import Link from "next/link";

interface CognitoUser {
  username: string;
  email: string;
  status: string;
  enabled: boolean;
  created_at: string;
  last_modified: string;
}

export default function AdminPage() {
  const [users, setUsers] = useState<CognitoUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [creating, setCreating] = useState(false);
  const [success, setSuccess] = useState("");

  const fetchUsers = async () => {
    try {
      const data = await apiGet<{ users: CognitoUser[] }>("/admin/users");
      setUsers(data.users || []);
    } catch (err) {
      if (err instanceof ApiAuthError) {
        signInWithRedirect();
        return;
      }
      setError("Failed to load users. You may not have admin permissions.");
    }
  };

  useEffect(() => {
    async function init() {
      try {
        await getCurrentUser();
      } catch {
        signInWithRedirect();
        return;
      }
      await fetchUsers();
      setLoading(false);
    }
    init();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newEmail.trim()) return;
    setCreating(true);
    setError("");
    setSuccess("");
    try {
      const res = await apiPost<{ message: string }>("/admin/users", {
        email: newEmail.trim(),
      });
      setSuccess(res.message);
      setNewEmail("");
      await fetchUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  };

  const handleToggle = async (username: string, enabled: boolean) => {
    try {
      const endpoint = enabled ? "disable" : "enable";
      await apiPost(`/admin/users/${encodeURIComponent(username)}/${endpoint}`, {});
      await fetchUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDelete = async (username: string, email: string) => {
    if (!confirm(`Permanently delete user ${email}? This cannot be undone.`)) return;
    try {
      await apiDelete(`/admin/users/${encodeURIComponent(username)}`);
      await fetchUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "CONFIRMED": return { bg: "#dcfce7", text: "#166534" };
      case "FORCE_CHANGE_PASSWORD": return { bg: "#fef3c7", text: "#92400e" };
      case "DISABLED": return { bg: "#fee2e2", text: "#991b1b" };
      default: return { bg: "#f3f4f6", text: "#4b5563" };
    }
  };

  if (loading) {
    return (
      <div style={{ padding: "4rem", textAlign: "center", color: "var(--muted)" }}>
        Loading...
      </div>
    );
  }

  return (
    <div style={{ maxWidth: "56rem", margin: "0 auto", padding: "2rem" }}>
      {/* Header */}
      <div style={{ marginBottom: "2rem", paddingBottom: "1rem", borderBottom: "1px solid var(--border)" }}>
        <Link
          href="/dashboard"
          style={{ color: "var(--muted)", textDecoration: "none", fontSize: "0.875rem" }}
        >
          &larr; Dashboard
        </Link>
        <h1 style={{ fontSize: "1.5rem", fontWeight: 700, marginTop: "0.5rem" }}>
          User Management
        </h1>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Add or remove users who can access Impulse. New users receive a temporary password via email.
        </p>
      </div>

      {/* Messages */}
      {error && (
        <div style={{ padding: "0.75rem 1rem", marginBottom: "1rem", backgroundColor: "#fef2f2", border: "1px solid #fecaca", borderRadius: "0.5rem", color: "#991b1b", fontSize: "0.85rem" }}>
          {error}
        </div>
      )}
      {success && (
        <div style={{ padding: "0.75rem 1rem", marginBottom: "1rem", backgroundColor: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: "0.5rem", color: "#166534", fontSize: "0.85rem" }}>
          {success}
        </div>
      )}

      {/* Add user form */}
      <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", padding: "1.25rem", marginBottom: "1.5rem", backgroundColor: "white" }}>
        <h2 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.75rem" }}>
          Add New User
        </h2>
        <form onSubmit={handleCreate} style={{ display: "flex", gap: "0.5rem" }}>
          <input
            type="email"
            value={newEmail}
            onChange={(e) => setNewEmail(e.target.value)}
            placeholder="user@northwestern.edu"
            required
            style={{
              flex: 1,
              padding: "0.5rem 0.75rem",
              border: "1px solid var(--border)",
              borderRadius: "0.375rem",
              fontSize: "0.875rem",
              outline: "none",
            }}
          />
          <button
            type="submit"
            disabled={!newEmail.trim() || creating}
            style={{
              padding: "0.5rem 1rem",
              fontSize: "0.8rem",
              fontWeight: 600,
              color: "white",
              backgroundColor: !newEmail.trim() || creating ? "#9ca3af" : "var(--primary)",
              border: "none",
              borderRadius: "0.375rem",
              cursor: !newEmail.trim() || creating ? "not-allowed" : "pointer",
              whiteSpace: "nowrap",
            }}
          >
            {creating ? "Adding..." : "Add User"}
          </button>
        </form>
      </div>

      {/* Users table */}
      <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", overflow: "hidden", backgroundColor: "white" }}>
        <div style={{ padding: "0.75rem 1rem", fontSize: "0.8rem", fontWeight: 600, color: "var(--muted)", borderBottom: "1px solid var(--border)", backgroundColor: "#fafafa", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Registered Users ({users.length})
        </div>
        {users.length === 0 ? (
          <div style={{ padding: "2rem", textAlign: "center", color: "var(--muted)", fontSize: "0.85rem" }}>
            No users registered yet.
          </div>
        ) : (
          <div>
            {users.map((user, i) => {
              const sc = statusColor(user.status);
              return (
                <div
                  key={user.username}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "0.75rem 1rem",
                    borderBottom: i < users.length - 1 ? "1px solid var(--border)" : "none",
                  }}
                >
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                      <span style={{ fontWeight: 500, fontSize: "0.875rem" }}>
                        {user.email || user.username}
                      </span>
                      <span
                        style={{
                          fontSize: "0.65rem",
                          fontWeight: 600,
                          padding: "0.0625rem 0.375rem",
                          borderRadius: "9999px",
                          backgroundColor: sc.bg,
                          color: sc.text,
                        }}
                      >
                        {user.status}
                      </span>
                      {!user.enabled && (
                        <span style={{ fontSize: "0.65rem", fontWeight: 600, padding: "0.0625rem 0.375rem", borderRadius: "9999px", backgroundColor: "#fee2e2", color: "#991b1b" }}>
                          DISABLED
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.125rem" }}>
                      Added {new Date(user.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: "0.375rem", flexShrink: 0 }}>
                    <button
                      onClick={() => handleToggle(user.username, user.enabled)}
                      style={{
                        padding: "0.25rem 0.5rem",
                        fontSize: "0.7rem",
                        fontWeight: 500,
                        color: user.enabled ? "var(--warning)" : "var(--success)",
                        backgroundColor: "transparent",
                        border: `1px solid ${user.enabled ? "var(--warning)" : "var(--success)"}`,
                        borderRadius: "0.25rem",
                        cursor: "pointer",
                      }}
                    >
                      {user.enabled ? "Disable" : "Enable"}
                    </button>
                    <button
                      onClick={() => handleDelete(user.username, user.email)}
                      style={{
                        padding: "0.25rem 0.5rem",
                        fontSize: "0.7rem",
                        fontWeight: 500,
                        color: "var(--danger)",
                        backgroundColor: "transparent",
                        border: "1px solid var(--danger)",
                        borderRadius: "0.25rem",
                        cursor: "pointer",
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
