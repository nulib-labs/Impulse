"use client";

import "@/lib/amplify-config";
import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getCurrentUser, signInWithRedirect } from "aws-amplify/auth";
import { apiGet, ApiAuthError } from "@/lib/api";
import { StatusBadge } from "@/components/status";
import Link from "next/link";

/* ── Types ──────────────────────────────────────────────────────────────── */

interface JobResult {
  job_id: string;
  custom_id?: string;
  status: string;
  task_type: string;
  ocr_engine?: string;
  metadata?: Record<string, string>;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  created_at: string;
  score: number;
}

interface DocumentResult {
  result_id: string;
  job_id: string;
  document_key: string;
  filename: string;
  page_number: number;
  extraction_model?: string;
  extracted_text?: string;
  summary?: string;
  created_at: string;
  score: number;
}

interface CollectionResult {
  collection_id: string;
  name: string;
  description?: string;
  document_count: number;
  created_at: string;
  score: number;
}

interface AnalysisResult {
  analysis_id: string;
  name: string;
  description?: string;
  status: string;
  sources?: { type: string; id: string; name: string }[];
  created_at: string;
  score: number;
}

interface SearchResponse {
  query: string;
  results: {
    jobs?: JobResult[];
    documents?: DocumentResult[];
    collections?: CollectionResult[];
    analyses?: AnalysisResult[];
  };
  counts: {
    jobs?: number;
    documents?: number;
    collections?: number;
    analyses?: number;
  };
  total: number;
  page: number;
  page_size: number;
}

type TabKey = "all" | "jobs" | "documents" | "collections" | "analyses";

/* ── Component ──────────────────────────────────────────────────────────── */

export default function SearchPage() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<TabKey>("all");

  // Read initial query from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q") || "";
    setQuery(q);
    setInputValue(q);
  }, []);

  // Fetch search results when query changes
  useEffect(() => {
    if (!query) {
      setLoading(false);
      setData(null);
      return;
    }

    let cancelled = false;

    async function doSearch() {
      setLoading(true);
      setError("");

      try {
        await getCurrentUser();
      } catch {
        signInWithRedirect();
        return;
      }

      try {
        const result = await apiGet<SearchResponse>(
          `/search?q=${encodeURIComponent(query)}`,
        );
        if (!cancelled) {
          setData(result);
        }
      } catch (err) {
        if (err instanceof ApiAuthError) {
          signInWithRedirect();
          return;
        }
        if (!cancelled) {
          setError(
            `Search failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      }

      if (!cancelled) setLoading(false);
    }

    doSearch();
    return () => {
      cancelled = true;
    };
  }, [query]);

  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const q = inputValue.trim();
      if (q && q !== query) {
        setQuery(q);
        // Update URL without full navigation
        window.history.pushState({}, "", `/search?q=${encodeURIComponent(q)}`);
      }
    },
    [inputValue, query],
  );

  const statusColor = (s: string) => {
    if (s === "COMPLETED") return { bg: "#dcfce7", text: "#166534" };
    if (s === "RUNNING") return { bg: "#dbeafe", text: "#1e40af" };
    if (s === "FAILED") return { bg: "#fee2e2", text: "#991b1b" };
    return { bg: "#f3f4f6", text: "#4b5563" };
  };

  const tabs: { key: TabKey; label: string; countKey?: keyof SearchResponse["counts"] }[] = [
    { key: "all", label: "All" },
    { key: "jobs", label: "Jobs", countKey: "jobs" },
    { key: "documents", label: "Documents", countKey: "documents" },
    { key: "collections", label: "Collections", countKey: "collections" },
    { key: "analyses", label: "Analyses", countKey: "analyses" },
  ];

  const hasResults = data && data.total > 0;
  const showJobs = (activeTab === "all" || activeTab === "jobs") && (data?.results.jobs?.length ?? 0) > 0;
  const showDocs = (activeTab === "all" || activeTab === "documents") && (data?.results.documents?.length ?? 0) > 0;
  const showCollections = (activeTab === "all" || activeTab === "collections") && (data?.results.collections?.length ?? 0) > 0;
  const showAnalyses = (activeTab === "all" || activeTab === "analyses") && (data?.results.analyses?.length ?? 0) > 0;

  return (
    <div style={{ maxWidth: "72rem", margin: "0 auto", padding: "2rem" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          marginBottom: "1.5rem",
          paddingBottom: "1rem",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <Link
          href="/dashboard"
          style={{
            color: "var(--muted)",
            textDecoration: "none",
            fontSize: "0.875rem",
            flexShrink: 0,
          }}
        >
          &larr; Dashboard
        </Link>

        {/* Search input */}
        <form onSubmit={handleSearch} style={{ flex: "1 1 auto", position: "relative" }}>
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
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Search jobs, documents, collections, analyses..."
            autoFocus
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
      </div>

      {/* Loading state */}
      {loading && (
        <div style={{ padding: "4rem", textAlign: "center", color: "var(--muted)" }}>
          Searching...
        </div>
      )}

      {/* Error state */}
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

      {/* No query state */}
      {!loading && !query && (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            border: "1px dashed var(--border)",
            borderRadius: "0.75rem",
            color: "var(--muted)",
          }}
        >
          <p>Enter a search term to find jobs, documents, collections, and analyses.</p>
        </div>
      )}

      {/* No results state */}
      {!loading && query && data && !hasResults && (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            border: "1px dashed var(--border)",
            borderRadius: "0.75rem",
            color: "var(--muted)",
          }}
        >
          <p style={{ marginBottom: "0.5rem" }}>
            No results found for &ldquo;{query}&rdquo;
          </p>
          <p style={{ fontSize: "0.8rem" }}>
            Try a different search term or check your spelling.
          </p>
        </div>
      )}

      {/* Results */}
      {!loading && hasResults && (
        <>
          {/* Summary */}
          <div
            style={{
              fontSize: "0.8rem",
              color: "var(--muted)",
              marginBottom: "1rem",
            }}
          >
            {data.total} result{data.total !== 1 ? "s" : ""} for &ldquo;{data.query}&rdquo;
          </div>

          {/* Tabs */}
          <div
            style={{
              display: "flex",
              gap: "0.25rem",
              marginBottom: "1.5rem",
              borderBottom: "1px solid var(--border)",
              overflowX: "auto",
            }}
          >
            {tabs.map((tab) => {
              const count = tab.key === "all"
                ? data.total
                : data.counts[tab.countKey!] ?? 0;
              if (tab.key !== "all" && count === 0) return null;
              const isActive = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    fontWeight: isActive ? 600 : 400,
                    color: isActive ? "var(--primary)" : "var(--muted)",
                    backgroundColor: "transparent",
                    border: "none",
                    borderBottom: isActive ? "2px solid var(--primary)" : "2px solid transparent",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                    marginBottom: "-1px",
                    transition: "color 0.15s",
                  }}
                >
                  {tab.label}
                  {count > 0 && (
                    <span
                      style={{
                        marginLeft: "0.375rem",
                        fontSize: "0.7rem",
                        padding: "0.0625rem 0.375rem",
                        backgroundColor: isActive ? "var(--accent)" : "#f3f4f6",
                        borderRadius: "9999px",
                      }}
                    >
                      {count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Job results */}
          {showJobs && (
            <ResultSection title="Jobs" count={data.counts.jobs ?? 0}>
              {data.results.jobs!.map((job) => (
                <Link
                  key={job.job_id}
                  href={`/jobs/${job.job_id}`}
                  style={cardStyle}
                  onMouseEnter={cardHoverIn}
                  onMouseLeave={cardHoverOut}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                        {job.custom_id && (
                          <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>
                            {job.custom_id}
                          </span>
                        )}
                        <span style={{ fontFamily: "monospace", fontSize: "0.75rem", color: "var(--muted)" }}>
                          {job.job_id.slice(0, 8)}...
                        </span>
                        <span
                          style={{
                            fontSize: "0.7rem",
                            color: "var(--muted)",
                            padding: "0.0625rem 0.375rem",
                            backgroundColor: "#f3f4f6",
                            borderRadius: "0.25rem",
                          }}
                        >
                          {job.task_type.replace(/_/g, " ")}
                        </span>
                      </div>
                    </div>
                    <StatusBadge status={job.status} />
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.375rem" }}>
                    {job.total_documents} document{job.total_documents !== 1 ? "s" : ""} &middot; Created {new Date(job.created_at).toLocaleDateString()}
                  </div>
                </Link>
              ))}
            </ResultSection>
          )}

          {/* Document results */}
          {showDocs && (
            <ResultSection title="Documents" count={data.counts.documents ?? 0}>
              {data.results.documents!.map((doc) => (
                <Link
                  key={doc.result_id}
                  href={`/jobs/${doc.job_id}`}
                  style={cardStyle}
                  onMouseEnter={cardHoverIn}
                  onMouseLeave={cardHoverOut}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>
                        {doc.filename}
                      </span>
                      {doc.page_number > 0 && (
                        <span style={{ fontSize: "0.7rem", color: "var(--muted)", marginLeft: "0.5rem" }}>
                          Page {doc.page_number}
                        </span>
                      )}
                    </div>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        fontFamily: "monospace",
                        color: "var(--muted)",
                        flexShrink: 0,
                      }}
                    >
                      {doc.job_id.slice(0, 8)}...
                    </span>
                  </div>
                  {doc.extracted_text && (
                    <p
                      style={{
                        fontSize: "0.8rem",
                        color: "var(--foreground)",
                        marginTop: "0.375rem",
                        lineHeight: 1.5,
                        opacity: 0.85,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {doc.extracted_text}
                    </p>
                  )}
                  {doc.summary && !doc.extracted_text && (
                    <p
                      style={{
                        fontSize: "0.8rem",
                        color: "var(--muted)",
                        marginTop: "0.375rem",
                        lineHeight: 1.5,
                        fontStyle: "italic",
                      }}
                    >
                      {doc.summary}
                    </p>
                  )}
                  <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.375rem" }}>
                    {doc.extraction_model && <>{doc.extraction_model} &middot; </>}
                    {new Date(doc.created_at).toLocaleDateString()}
                  </div>
                </Link>
              ))}
            </ResultSection>
          )}

          {/* Collection results */}
          {showCollections && (
            <ResultSection title="Collections" count={data.counts.collections ?? 0}>
              {data.results.collections!.map((col) => (
                <Link
                  key={col.collection_id}
                  href={`/collections/${col.collection_id}`}
                  style={cardStyle}
                  onMouseEnter={cardHoverIn}
                  onMouseLeave={cardHoverOut}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <h3 style={{ fontSize: "0.95rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                        {col.name}
                      </h3>
                      {col.description && (
                        <p style={{ fontSize: "0.8rem", color: "var(--muted)" }}>
                          {col.description}
                        </p>
                      )}
                    </div>
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 500,
                        padding: "0.125rem 0.5rem",
                        backgroundColor: "#f3f4f6",
                        borderRadius: "9999px",
                        color: "var(--muted)",
                        whiteSpace: "nowrap",
                        flexShrink: 0,
                      }}
                    >
                      {col.document_count} doc{col.document_count !== 1 ? "s" : ""}
                    </span>
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.375rem" }}>
                    Created {new Date(col.created_at).toLocaleDateString()}
                  </div>
                </Link>
              ))}
            </ResultSection>
          )}

          {/* Analysis results */}
          {showAnalyses && (
            <ResultSection title="Analyses" count={data.counts.analyses ?? 0}>
              {data.results.analyses!.map((a) => {
                const sc = statusColor(a.status);
                return (
                  <Link
                    key={a.analysis_id}
                    href={`/analyses/${a.analysis_id}`}
                    style={cardStyle}
                    onMouseEnter={cardHoverIn}
                    onMouseLeave={cardHoverOut}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div>
                        <h3 style={{ fontSize: "0.95rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                          {a.name}
                        </h3>
                        {a.description && (
                          <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.25rem" }}>
                            {a.description}
                          </p>
                        )}
                        {a.sources && a.sources.length > 0 && (
                          <div style={{ display: "flex", gap: "0.25rem", flexWrap: "wrap" }}>
                            {a.sources.slice(0, 3).map((s, i) => (
                              <span
                                key={i}
                                style={{
                                  fontSize: "0.65rem",
                                  padding: "0.0625rem 0.375rem",
                                  backgroundColor: "#f3f4f6",
                                  borderRadius: "0.25rem",
                                  color: "var(--muted)",
                                }}
                              >
                                {s.name}
                              </span>
                            ))}
                            {a.sources.length > 3 && (
                              <span style={{ fontSize: "0.65rem", color: "var(--muted)" }}>
                                +{a.sources.length - 3} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                      <span
                        style={{
                          fontSize: "0.7rem",
                          fontWeight: 600,
                          padding: "0.125rem 0.5rem",
                          borderRadius: "9999px",
                          backgroundColor: sc.bg,
                          color: sc.text,
                          whiteSpace: "nowrap",
                          flexShrink: 0,
                        }}
                      >
                        {a.status}
                      </span>
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.375rem" }}>
                      Created {new Date(a.created_at).toLocaleDateString()}
                    </div>
                  </Link>
                );
              })}
            </ResultSection>
          )}
        </>
      )}
    </div>
  );
}

/* ── Shared styles & helpers ────────────────────────────────────────────── */

const cardStyle: React.CSSProperties = {
  display: "block",
  padding: "1rem 1.25rem",
  border: "1px solid var(--border)",
  borderRadius: "0.75rem",
  textDecoration: "none",
  color: "inherit",
  backgroundColor: "white",
  transition: "border-color 0.15s, box-shadow 0.15s",
};

const cardHoverIn = (e: React.MouseEvent<HTMLAnchorElement>) => {
  e.currentTarget.style.borderColor = "var(--primary)";
  e.currentTarget.style.boxShadow = "0 1px 3px rgba(37, 99, 235, 0.1)";
};

const cardHoverOut = (e: React.MouseEvent<HTMLAnchorElement>) => {
  e.currentTarget.style.borderColor = "var(--border)";
  e.currentTarget.style.boxShadow = "none";
};

function ResultSection({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: "1.5rem" }}>
      <h2
        style={{
          fontSize: "0.85rem",
          fontWeight: 600,
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: "0.75rem",
        }}
      >
        {title}
        <span
          style={{
            marginLeft: "0.5rem",
            fontSize: "0.75rem",
            fontWeight: 400,
            textTransform: "none",
            letterSpacing: "normal",
          }}
        >
          ({count})
        </span>
      </h2>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {children}
      </div>
    </div>
  );
}
