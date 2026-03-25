"use client";

import "@/lib/amplify-config";
import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getCurrentUser, signOut, signInWithRedirect } from "aws-amplify/auth";
import { apiGet, ApiAuthError } from "@/lib/api";
import { StatusBadge, ProgressBar, MetadataTags } from "@/components/status";
import Link from "next/link";

interface Job {
  job_id: string;
  status: string;
  task_type: string;
  ocr_engine?: string;
  custom_id?: string;
  metadata?: Record<string, string>;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  created_at: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [username, setUsername] = useState("");
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const q = searchQuery.trim();
      if (q) {
        router.push(`/search?q=${encodeURIComponent(q)}`);
      }
    },
    [searchQuery, router],
  );

  useEffect(() => {
    let cancelled = false;

    async function init() {
      // Step 1: Check auth
      let user;
      try {
        user = await getCurrentUser();
      } catch {
        signInWithRedirect();
        return;
      }

      if (cancelled) return;
      setUsername(user.signInDetails?.loginId || user.username);

      // Step 2: Fetch jobs
      try {
        const data = await apiGet<{ jobs: Job[] }>("/jobs");
        if (!cancelled) {
          setJobs(data.jobs || []);
        }
      } catch (err) {
        console.error("Failed to fetch jobs:", err);
        if (!cancelled) {
          if (err instanceof ApiAuthError) {
            signInWithRedirect();
            return;
          }
          setError("Failed to load jobs. The API may not be connected yet.");
        }
      }

      if (!cancelled) setLoading(false);
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSignOut = async () => {
    await signOut();
    router.replace("/");
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
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "2rem",
          paddingBottom: "1rem",
          borderBottom: "1px solid var(--border)",
          gap: "1rem",
        }}
      >
        <div style={{ flexShrink: 0 }}>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 700 }}>Impulse</h1>
          <p style={{ color: "var(--muted)", fontSize: "0.8rem", marginTop: "0.125rem" }}>
            Signed in as {username}
          </p>
        </div>

        {/* Search bar */}
        <form
          onSubmit={handleSearch}
          style={{
            flex: "1 1 auto",
            maxWidth: "28rem",
            position: "relative",
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--muted)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{
              position: "absolute",
              left: "0.75rem",
              top: "50%",
              transform: "translateY(-50%)",
              pointerEvents: "none",
            }}
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search jobs, documents, collections..."
            style={{
              width: "100%",
              padding: "0.5rem 0.75rem 0.5rem 2.25rem",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
              fontSize: "0.875rem",
              backgroundColor: "var(--surface)",
              color: "var(--foreground)",
              outline: "none",
              boxSizing: "border-box",
              transition: "border-color 0.15s",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--primary)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--border)";
            }}
          />
        </form>

        <div style={{ display: "flex", gap: "0.75rem", flexShrink: 0 }}>
          <Link
            href="/admin"
            style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "0.5rem 1rem",
              backgroundColor: "transparent",
              color: "var(--muted)",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
              textDecoration: "none",
              fontSize: "0.875rem",
              fontWeight: 500,
            }}
          >
            Admin
          </Link>
          <Link
            href="/analyses"
            style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "0.5rem 1rem",
              backgroundColor: "transparent",
              color: "var(--foreground)",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
              textDecoration: "none",
              fontSize: "0.875rem",
              fontWeight: 500,
            }}
          >
            Analysis
          </Link>
          <Link
            href="/collections"
            style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "0.5rem 1rem",
              backgroundColor: "transparent",
              color: "var(--foreground)",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
              textDecoration: "none",
              fontSize: "0.875rem",
              fontWeight: 500,
            }}
          >
            Collections
          </Link>
          <Link
            href="/jobs/new"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.375rem",
              padding: "0.5rem 1rem",
              backgroundColor: "var(--primary)",
              color: "white",
              borderRadius: "0.5rem",
              textDecoration: "none",
              fontSize: "0.875rem",
              fontWeight: 500,
              transition: "background-color 0.15s",
            }}
          >
            + New Job
          </Link>
          <button
            onClick={handleSignOut}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "transparent",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
              cursor: "pointer",
              fontSize: "0.875rem",
              color: "var(--muted)",
            }}
          >
            Sign Out
          </button>
        </div>
      </header>

      {/* Error banner */}
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

      {/* Jobs list */}
      <h2 style={{ fontSize: "1.125rem", fontWeight: 600, marginBottom: "1rem" }}>
        Your Jobs ({jobs.length})
      </h2>

      {jobs.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            border: "1px dashed var(--border)",
            borderRadius: "0.75rem",
            color: "var(--muted)",
          }}
        >
          <div style={{ fontSize: "1.5rem", marginBottom: "0.75rem", opacity: 0.3 }}>
            &#128196;
          </div>
          <p style={{ marginBottom: "0.5rem" }}>No jobs yet.</p>
          <Link
            href="/jobs/new"
            style={{ color: "var(--primary)", textDecoration: "underline", fontSize: "0.875rem" }}
          >
            Create your first job
          </Link>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {jobs.map((job) => (
            <Link
              key={job.job_id}
              href={`/jobs/${job.job_id}`}
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
                e.currentTarget.style.boxShadow = "0 1px 3px rgba(37, 99, 235, 0.1)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border)";
                e.currentTarget.style.boxShadow = "none";
              }}
            >
              {/* Top row: title + status */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  marginBottom: "0.5rem",
                }}
              >
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                    {job.custom_id && (
                      <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>
                        {job.custom_id}
                      </span>
                    )}
                    <span
                      style={{
                        fontFamily: "monospace",
                        fontSize: "0.75rem",
                        color: "var(--muted)",
                      }}
                    >
                      {job.job_id.slice(0, 8)}...
                    </span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--muted)",
                        padding: "0.0625rem 0.375rem",
                        backgroundColor: "#f3f4f6",
                        borderRadius: "0.25rem",
                      }}
                    >
                      {job.task_type.replace(/_/g, " ")}
                    </span>
                    {job.ocr_engine && (
                      <span
                        style={{
                          fontSize: "0.75rem",
                          color: "var(--muted)",
                          padding: "0.0625rem 0.375rem",
                          backgroundColor: "#eff6ff",
                          borderRadius: "0.25rem",
                        }}
                      >
                        {job.ocr_engine === "textract" ? "Textract" : job.ocr_engine === "bedrock_claude" ? "Claude" : "Marker"}
                      </span>
                    )}
                  </div>
                </div>
                <StatusBadge status={job.status} />
              </div>

              {/* Metadata tags */}
              {job.metadata && Object.keys(job.metadata).length > 0 && (
                <MetadataTags metadata={job.metadata} style={{ marginBottom: "0.75rem" }} />
              )}

              {/* Progress */}
              <ProgressBar
                processed={job.processed_documents}
                failed={job.failed_documents}
                total={job.total_documents}
              />

              {/* Footer */}
              <div
                style={{
                  marginTop: "0.5rem",
                  fontSize: "0.7rem",
                  color: "var(--muted)",
                }}
              >
                Created {new Date(job.created_at).toLocaleString()}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
