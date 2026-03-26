"use client";

/* ── StatusBadge ─────────────────────────────────────────────────────── */

interface StatusBadgeProps {
  status: string;
}

const STATUS_CONFIG: Record<string, { bg: string; text: string; dot?: string }> = {
  PENDING:    { bg: "background-color: #fef3c7", text: "color: #92400e" },
  PROCESSING: { bg: "background-color: #dbeafe", text: "color: #1e40af", dot: "#3b82f6" },
  COMPLETED:  { bg: "background-color: #dcfce7", text: "color: #166534" },
  FAILED:     { bg: "background-color: #fee2e2", text: "color: #991b1b" },
};

export function StatusBadge({ status }: StatusBadgeProps) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.PENDING;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.375rem",
        padding: "0.25rem 0.625rem",
        borderRadius: "9999px",
        fontSize: "0.75rem",
        fontWeight: 600,
        ...Object.fromEntries(
          [cfg.bg, cfg.text].map((s) => {
            const [k, v] = s.split(": ");
            return [k.replace(/-([a-z])/g, (_, l) => l.toUpperCase()), v];
          })
        ),
      }}
    >
      {cfg.dot && (
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            backgroundColor: cfg.dot,
            animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
          }}
        />
      )}
      {status}
    </span>
  );
}

/* ── ProgressBar ─────────────────────────────────────────────────────── */

interface ProgressBarProps {
  processed: number;
  failed: number;
  total: number;
}

export function ProgressBar({ processed, failed, total }: ProgressBarProps) {
  const pct = total > 0 ? Math.round(((processed + failed) / total) * 100) : 0;
  const successPct = total > 0 ? Math.round((processed / total) * 100) : 0;
  const failPct = total > 0 ? Math.round((failed / total) * 100) : 0;

  return (
    <div style={{ width: "100%" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "0.75rem",
          color: "var(--muted)",
          marginBottom: "0.25rem",
        }}
      >
        <span>
          {processed} / {total} processed
          {failed > 0 && (
            <span style={{ color: "var(--danger)", marginLeft: "0.25rem" }}>
              ({failed} failed)
            </span>
          )}
        </span>
        <span style={{ fontWeight: 500 }}>{pct}%</span>
      </div>
      <div
        style={{
          width: "100%",
          height: "0.5rem",
          backgroundColor: "var(--border)",
          borderRadius: "0.25rem",
          overflow: "hidden",
          display: "flex",
        }}
      >
        <div
          style={{
            width: `${successPct}%`,
            height: "100%",
            backgroundColor: "var(--success)",
            transition: "width 0.5s ease-out",
          }}
        />
        {failPct > 0 && (
          <div
            style={{
              width: `${failPct}%`,
              height: "100%",
              backgroundColor: "var(--danger)",
              transition: "width 0.5s ease-out",
            }}
          />
        )}
      </div>
    </div>
  );
}

/* ── MetadataTags ────────────────────────────────────────────────────── */

interface MetadataTagsProps {
  metadata: Record<string, string>;
  style?: React.CSSProperties;
}

export function MetadataTags({ metadata, style }: MetadataTagsProps) {
  const entries = Object.entries(metadata || {});
  if (entries.length === 0) return null;

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem", ...style }}>
      {entries.map(([key, value]) => (
        <span
          key={key}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.25rem",
            padding: "0.125rem 0.5rem",
            borderRadius: "0.25rem",
            fontSize: "0.7rem",
            backgroundColor: "#f3f4f6",
            color: "#4b5563",
          }}
        >
          <span style={{ fontWeight: 600, color: "#1f2937" }}>{key}:</span>
          <span>{value}</span>
        </span>
      ))}
    </div>
  );
}
