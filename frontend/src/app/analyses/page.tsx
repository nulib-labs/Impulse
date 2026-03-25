"use client";

import "@/lib/amplify-config";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getCurrentUser, signInWithRedirect } from "aws-amplify/auth";
import { apiGet, apiPost, ApiAuthError } from "@/lib/api";
import Link from "next/link";

interface AnalysisSummary {
  analysis_id: string;
  name: string;
  description: string;
  status: string;
  source_count: number;
  sources: { type: string; id: string; name: string }[];
  created_at: string;
  updated_at: string;
}

interface JobOption {
  job_id: string;
  custom_id?: string;
  status: string;
  task_type: string;
}

interface CollectionOption {
  collection_id: string;
  name: string;
  document_count: number;
}

export default function AnalysesPage() {
  const router = useRouter();
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [jobs, setJobs] = useState<JobOption[]>([]);
  const [collections, setCollections] = useState<CollectionOption[]>([]);
  const [selectedSources, setSelectedSources] = useState<{ type: string; id: string; name: string }[]>([]);

  useEffect(() => {
    async function init() {
      try { await getCurrentUser(); } catch { signInWithRedirect(); return; }
      try {
        const [aData, jData, cData] = await Promise.all([
          apiGet<{ analyses: AnalysisSummary[] }>("/analyses"),
          apiGet<{ jobs: JobOption[] }>("/jobs"),
          apiGet<{ collections: CollectionOption[] }>("/collections"),
        ]);
        setAnalyses(aData.analyses || []);
        setJobs(jData.jobs || []);
        setCollections(cData.collections || []);
      } catch (err) {
        if (err instanceof ApiAuthError) { signInWithRedirect(); return; }
        setError("Failed to load data.");
      }
      setLoading(false);
    }
    init();
  }, []);

  const toggleSource = (type: string, id: string, name: string) => {
    setSelectedSources((prev) => {
      const exists = prev.find((s) => s.id === id);
      if (exists) return prev.filter((s) => s.id !== id);
      return [...prev, { type, id, name }];
    });
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || selectedSources.length === 0) return;
    setCreating(true);
    try {
      const res = await apiPost<{ analysis_id: string }>("/analyses", {
        name: newName.trim(),
        description: newDesc.trim(),
        sources: selectedSources,
      });
      router.push(`/analyses/${res.analysis_id}`);
    } catch (err) {
      setError(`Failed to create: ${err instanceof Error ? err.message : String(err)}`);
      setCreating(false);
    }
  };

  const statusColor = (s: string) => {
    if (s === "COMPLETED") return { bg: "#dcfce7", text: "#166534" };
    if (s === "RUNNING") return { bg: "#dbeafe", text: "#1e40af" };
    if (s === "FAILED") return { bg: "#fee2e2", text: "#991b1b" };
    return { bg: "#f3f4f6", text: "#4b5563" };
  };

  if (loading) return <div style={{ padding: "4rem", textAlign: "center", color: "var(--muted)" }}>Loading...</div>;

  return (
    <div style={{ maxWidth: "72rem", margin: "0 auto", padding: "2rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem", paddingBottom: "1rem", borderBottom: "1px solid var(--border)" }}>
        <div>
          <Link href="/dashboard" style={{ color: "var(--muted)", textDecoration: "none", fontSize: "0.875rem" }}>&larr; Dashboard</Link>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 700, marginTop: "0.5rem" }}>Analyses</h1>
        </div>
        <button onClick={() => { setShowCreate(true); setSelectedSources([]); }} style={{ padding: "0.5rem 1rem", backgroundColor: "var(--primary)", color: "white", border: "none", borderRadius: "0.5rem", fontWeight: 500, fontSize: "0.875rem", cursor: "pointer" }}>
          + New Analysis
        </button>
      </div>

      {error && <div style={{ padding: "0.75rem 1rem", marginBottom: "1rem", backgroundColor: "#fef2f2", border: "1px solid #fecaca", borderRadius: "0.5rem", color: "#991b1b", fontSize: "0.85rem" }}>{error}</div>}

      {/* Create modal */}
      {showCreate && (
        <div onClick={() => setShowCreate(false)} style={{ position: "fixed", inset: 0, backgroundColor: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50, padding: "2rem" }}>
          <div onClick={(e) => e.stopPropagation()} style={{ backgroundColor: "white", borderRadius: "0.75rem", padding: "1.5rem", maxWidth: "36rem", width: "100%", maxHeight: "80vh", overflowY: "auto", boxShadow: "0 25px 50px rgba(0,0,0,0.25)" }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "1rem" }}>Create Analysis</h3>
            <form onSubmit={handleCreate}>
              <div style={{ marginBottom: "0.75rem" }}>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.25rem" }}>Name</label>
                <input type="text" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. Africana Collection Analysis" autoFocus style={{ width: "100%", padding: "0.5rem 0.75rem", border: "1px solid var(--border)", borderRadius: "0.375rem", fontSize: "0.875rem", boxSizing: "border-box", outline: "none" }} />
              </div>
              <div style={{ marginBottom: "1rem" }}>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.25rem" }}>Description <span style={{ fontWeight: 400, color: "var(--muted)" }}>(optional)</span></label>
                <textarea value={newDesc} onChange={(e) => setNewDesc(e.target.value)} rows={2} placeholder="What are you analyzing?" style={{ width: "100%", padding: "0.5rem 0.75rem", border: "1px solid var(--border)", borderRadius: "0.375rem", fontSize: "0.875rem", boxSizing: "border-box", outline: "none", resize: "vertical" }} />
              </div>

              {/* Source picker */}
              <div style={{ marginBottom: "1rem" }}>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.5rem" }}>
                  Select Sources ({selectedSources.length} selected)
                </label>

                {jobs.length > 0 && (
                  <div style={{ marginBottom: "0.75rem" }}>
                    <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.375rem" }}>Jobs</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", maxHeight: "10rem", overflowY: "auto" }}>
                      {jobs.filter(j => j.status === "COMPLETED").map((j) => {
                        const selected = selectedSources.some((s) => s.id === j.job_id);
                        return (
                          <label key={j.job_id} style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.375rem 0.5rem", borderRadius: "0.25rem", cursor: "pointer", backgroundColor: selected ? "#eff6ff" : "transparent", fontSize: "0.8rem" }}>
                            <input type="checkbox" checked={selected} onChange={() => toggleSource("job", j.job_id, j.custom_id || j.job_id.slice(0, 8))} style={{ accentColor: "var(--primary)" }} />
                            <span style={{ fontWeight: 500 }}>{j.custom_id || j.job_id.slice(0, 8)}</span>
                            <span style={{ color: "var(--muted)", fontSize: "0.7rem" }}>{j.task_type.replace(/_/g, " ")}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                )}

                {collections.length > 0 && (
                  <div>
                    <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.375rem" }}>Collections</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", maxHeight: "10rem", overflowY: "auto" }}>
                      {collections.map((c) => {
                        const selected = selectedSources.some((s) => s.id === c.collection_id);
                        return (
                          <label key={c.collection_id} style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.375rem 0.5rem", borderRadius: "0.25rem", cursor: "pointer", backgroundColor: selected ? "#eff6ff" : "transparent", fontSize: "0.8rem" }}>
                            <input type="checkbox" checked={selected} onChange={() => toggleSource("collection", c.collection_id, c.name)} style={{ accentColor: "var(--primary)" }} />
                            <span style={{ fontWeight: 500 }}>{c.name}</span>
                            <span style={{ color: "var(--muted)", fontSize: "0.7rem" }}>{c.document_count} docs</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
                <button type="button" onClick={() => setShowCreate(false)} style={{ padding: "0.5rem 1rem", fontSize: "0.8rem", backgroundColor: "transparent", border: "1px solid var(--border)", borderRadius: "0.375rem", cursor: "pointer" }}>Cancel</button>
                <button type="submit" disabled={!newName.trim() || selectedSources.length === 0 || creating} style={{ padding: "0.5rem 1rem", fontSize: "0.8rem", fontWeight: 600, color: "white", backgroundColor: (!newName.trim() || selectedSources.length === 0 || creating) ? "#9ca3af" : "var(--primary)", border: "none", borderRadius: "0.375rem", cursor: (!newName.trim() || selectedSources.length === 0 || creating) ? "not-allowed" : "pointer" }}>
                  {creating ? "Creating..." : "Create & Analyze"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Analyses list */}
      {analyses.length === 0 ? (
        <div style={{ textAlign: "center", padding: "3rem", border: "1px dashed var(--border)", borderRadius: "0.75rem", color: "var(--muted)" }}>
          <p style={{ marginBottom: "0.5rem" }}>No analyses yet.</p>
          <button onClick={() => setShowCreate(true)} style={{ color: "var(--primary)", backgroundColor: "transparent", border: "none", cursor: "pointer", textDecoration: "underline", fontSize: "0.875rem" }}>Create your first analysis</button>
        </div>
      ) : (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {analyses.map((a) => {
            const sc = statusColor(a.status);
            return (
              <Link key={a.analysis_id} href={`/analyses/${a.analysis_id}`} style={{ display: "block", padding: "1.25rem", border: "1px solid var(--border)", borderRadius: "0.75rem", textDecoration: "none", color: "inherit", backgroundColor: "white", transition: "border-color 0.15s" }} onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--primary)"; }} onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.25rem" }}>{a.name}</h3>
                    {a.description && <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.375rem" }}>{a.description}</p>}
                    <div style={{ display: "flex", gap: "0.375rem", flexWrap: "wrap" }}>
                      {a.sources?.slice(0, 4).map((s, i) => (
                        <span key={i} style={{ fontSize: "0.65rem", padding: "0.0625rem 0.375rem", backgroundColor: "#f3f4f6", borderRadius: "0.25rem", color: "var(--muted)" }}>{s.name}</span>
                      ))}
                      {(a.sources?.length || 0) > 4 && <span style={{ fontSize: "0.65rem", color: "var(--muted)" }}>+{a.sources.length - 4} more</span>}
                    </div>
                  </div>
                  <span style={{ fontSize: "0.7rem", fontWeight: 600, padding: "0.125rem 0.5rem", borderRadius: "9999px", backgroundColor: sc.bg, color: sc.text, whiteSpace: "nowrap" }}>{a.status}</span>
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.5rem" }}>Created {new Date(a.created_at).toLocaleDateString()}</div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
