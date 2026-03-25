"use client";

import "@/lib/amplify-config";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getCurrentUser, signInWithRedirect } from "aws-amplify/auth";
import { apiGet, apiPost, apiDelete, ApiAuthError } from "@/lib/api";
import Link from "next/link";

interface CollectionSummary {
  collection_id: string;
  name: string;
  description: string;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export default function CollectionsPage() {
  const router = useRouter();
  const [collections, setCollections] = useState<CollectionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);

  const fetchCollections = async () => {
    try {
      const data = await apiGet<{ collections: CollectionSummary[] }>("/collections");
      setCollections(data.collections || []);
    } catch (err) {
      if (err instanceof ApiAuthError) {
        signInWithRedirect();
        return;
      }
      setError("Failed to load collections.");
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
      await fetchCollections();
      setLoading(false);
    }
    init();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const res = await apiPost<{ collection_id: string }>("/collections", {
        name: newName.trim(),
        description: newDesc.trim(),
      });
      router.push(`/collections/${res.collection_id}`);
    } catch (err) {
      setError(`Failed to create: ${err instanceof Error ? err.message : String(err)}`);
      setCreating(false);
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
    <div style={{ maxWidth: "72rem", margin: "0 auto", padding: "2rem" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "2rem",
          paddingBottom: "1rem",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <Link
              href="/dashboard"
              style={{ color: "var(--muted)", textDecoration: "none", fontSize: "0.875rem" }}
            >
              &larr; Dashboard
            </Link>
          </div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 700, marginTop: "0.5rem" }}>Collections</h1>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: "var(--primary)",
            color: "white",
            border: "none",
            borderRadius: "0.5rem",
            fontWeight: 500,
            fontSize: "0.875rem",
            cursor: "pointer",
          }}
        >
          + New Collection
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: "0.75rem 1rem",
            marginBottom: "1rem",
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: "0.5rem",
            color: "#991b1b",
            fontSize: "0.875rem",
          }}
        >
          {error}
        </div>
      )}

      {/* Create modal */}
      {showCreate && (
        <div
          onClick={() => setShowCreate(false)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 50,
            padding: "2rem",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              padding: "1.5rem",
              maxWidth: "28rem",
              width: "100%",
              boxShadow: "0 25px 50px rgba(0,0,0,0.25)",
            }}
          >
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "1rem" }}>
              Create Collection
            </h3>
            <form onSubmit={handleCreate}>
              <div style={{ marginBottom: "0.75rem" }}>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Name
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Africana Collection 2026"
                  autoFocus
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem",
                    boxSizing: "border-box",
                    outline: "none",
                  }}
                />
              </div>
              <div style={{ marginBottom: "1rem" }}>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Description <span style={{ fontWeight: 400, color: "var(--muted)" }}>(optional)</span>
                </label>
                <textarea
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  placeholder="What is this collection for?"
                  rows={2}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem",
                    boxSizing: "border-box",
                    outline: "none",
                    resize: "vertical",
                  }}
                />
              </div>
              <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
                <button
                  type="button"
                  onClick={() => setShowCreate(false)}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    backgroundColor: "transparent",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    cursor: "pointer",
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!newName.trim() || creating}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    fontWeight: 600,
                    color: "white",
                    backgroundColor: !newName.trim() || creating ? "#9ca3af" : "var(--primary)",
                    border: "none",
                    borderRadius: "0.375rem",
                    cursor: !newName.trim() || creating ? "not-allowed" : "pointer",
                  }}
                >
                  {creating ? "Creating..." : "Create"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Collections list */}
      {collections.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            border: "1px dashed var(--border)",
            borderRadius: "0.75rem",
            color: "var(--muted)",
          }}
        >
          <p style={{ marginBottom: "0.5rem" }}>No collections yet.</p>
          <button
            onClick={() => setShowCreate(true)}
            style={{
              color: "var(--primary)",
              backgroundColor: "transparent",
              border: "none",
              cursor: "pointer",
              textDecoration: "underline",
              fontSize: "0.875rem",
            }}
          >
            Create your first collection
          </button>
        </div>
      ) : (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {collections.map((col) => (
            <Link
              key={col.collection_id}
              href={`/collections/${col.collection_id}`}
              style={{
                display: "block",
                padding: "1.25rem",
                border: "1px solid var(--border)",
                borderRadius: "0.75rem",
                textDecoration: "none",
                color: "inherit",
                backgroundColor: "white",
                transition: "border-color 0.15s, box-shadow 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "var(--primary)";
                e.currentTarget.style.boxShadow = "0 1px 3px rgba(37,99,235,0.1)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border)";
                e.currentTarget.style.boxShadow = "none";
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                    {col.name}
                  </h3>
                  {col.description && (
                    <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
                      {col.description}
                    </p>
                  )}
                </div>
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 500,
                    padding: "0.125rem 0.5rem",
                    backgroundColor: "#f3f4f6",
                    borderRadius: "9999px",
                    color: "var(--muted)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {col.document_count} doc{col.document_count !== 1 ? "s" : ""}
                </span>
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--muted)" }}>
                Created {new Date(col.created_at).toLocaleDateString()}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
