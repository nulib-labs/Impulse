"use client";

import "@/lib/amplify-config";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiGet, apiPost } from "@/lib/api";
import Link from "next/link";

const TASK_TYPES = [
  { value: "full_pipeline", label: "Full Pipeline", desc: "Image processing + OCR + metadata extraction" },
  { value: "image_transform", label: "Image Transformation", desc: "Grayscale, binarize, denoise, encode" },
  { value: "document_extraction", label: "Document Extraction", desc: "OCR text extraction from documents" },
  { value: "metadata_extraction", label: "Metadata Extraction", desc: "NER + Bedrock LLM analysis" },
  { value: "mets_conversion", label: "METS Conversion", desc: "METS XML to HathiTrust YAML" },
  { value: "summarisation", label: "Summarisation", desc: "Document summary via Bedrock" },
  { value: "ner", label: "Named Entity Recognition", desc: "BERT-based NER extraction" },
];

const OCR_ENGINES = [
  {
    value: "textract",
    label: "AWS Textract",
    desc: "Fast cloud OCR. Best for printed documents, forms, and tables.",
    tag: "Recommended",
  },
  {
    value: "bedrock_claude",
    label: "Claude Vision",
    desc: "AI-powered OCR. Best for handwritten text and degraded historical documents.",
    tag: "Best for handwriting",
  },
  {
    value: "marker_pdf",
    label: "Marker PDF",
    desc: "Local ML pipeline. Best for complex PDF layouts with tables and figures. Slower.",
    tag: "PDF specialist",
  },
];

interface MetadataEntry {
  id: string;
  key: string;
  value: string;
}

interface CollectionOption {
  collection_id: string;
  name: string;
  document_count: number;
}

let entryCounter = 0;
function nextId() {
  return `entry-${++entryCounter}`;
}

export default function NewJobPage() {
  const router = useRouter();
  const [taskType, setTaskType] = useState("full_pipeline");
  const [ocrEngine, setOcrEngine] = useState("textract");
  const [customId, setCustomId] = useState("");
  const [metadataEntries, setMetadataEntries] = useState<MetadataEntry[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState("");
  const [dragActive, setDragActive] = useState(false);

  // ── Collection picker state ───────────────────────────────────
  const [availableCollections, setAvailableCollections] = useState<CollectionOption[]>([]);
  const [selectedCollectionIds, setSelectedCollectionIds] = useState<Set<string>>(new Set());
  const [loadingCollections, setLoadingCollections] = useState(true);
  const [showCreateCollection, setShowCreateCollection] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [newCollectionDescription, setNewCollectionDescription] = useState("");
  const [creatingCollection, setCreatingCollection] = useState(false);

  const addMetadataEntry = () => {
    setMetadataEntries((prev) => [...prev, { id: nextId(), key: "", value: "" }]);
  };

  const updateMetadataEntry = (id: string, field: "key" | "value", val: string) => {
    setMetadataEntries((prev) =>
      prev.map((e) => (e.id === id ? { ...e, [field]: val } : e))
    );
  };

  const removeMetadataEntry = (id: string) => {
    setMetadataEntries((prev) => prev.filter((e) => e.id !== id));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files.length > 0) {
      setFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // ── Fetch available collections on mount ──────────────────────
  useEffect(() => {
    (async () => {
      try {
        const data = await apiGet<{ collections: CollectionOption[] }>("/collections");
        setAvailableCollections(data.collections || []);
      } catch { /* ignore -- user may not have any collections */ }
      setLoadingCollections(false);
    })();
  }, []);

  const toggleCollection = (id: string) => {
    setSelectedCollectionIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleCreateCollection = async () => {
    const name = newCollectionName.trim();
    if (!name) return;
    setCreatingCollection(true);
    try {
      const res = await apiPost<{ collection_id: string }>("/collections", {
        name,
        description: newCollectionDescription.trim() || undefined,
      });
      const created: CollectionOption = {
        collection_id: res.collection_id,
        name,
        document_count: 0,
      };
      setAvailableCollections((prev) => [...prev, created]);
      setSelectedCollectionIds((prev) => new Set(prev).add(created.collection_id));
      setNewCollectionName("");
      setNewCollectionDescription("");
      setShowCreateCollection(false);
    } catch (err) {
      alert(`Failed to create collection: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCreatingCollection(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;

    setUploading(true);
    try {
      // Build metadata dict from entries
      const metadata: Record<string, string> = {};
      for (const entry of metadataEntries) {
        const k = entry.key.trim();
        const v = entry.value.trim();
        if (k && v) metadata[k] = v;
      }

      // Step 1: Create the job
      setProgress("Creating job...");
      const jobRes = await apiPost<{
        job_id: string;
        input_s3_prefix: string;
      }>("/jobs", {
        task_type: taskType,
        file_count: files.length,
        ocr_engine: ocrEngine,
        custom_id: customId.trim() || undefined,
        metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
      });

      const jobId = jobRes.job_id;

      // Step 2: Get presigned upload URLs (do NOT start processing yet)
      setProgress("Generating upload URLs...");
      const urlRes = await apiPost<{
        upload_urls: Record<string, string>;
      }>(`/jobs/${jobId}/upload-url`, {
        filenames: files.map((f) => f.name),
        start_processing: false,
      });

      // Step 3: Upload files directly to S3 via presigned URLs
      const uploadUrls = urlRes.upload_urls;
      let uploaded = 0;

      for (const file of files) {
        setProgress(`Uploading ${file.name} (${uploaded + 1}/${files.length})...`);
        const url = uploadUrls[file.name];
        if (!url) continue;

        await fetch(url, {
          method: "PUT",
          body: file,
          headers: { "Content-Type": "application/octet-stream" },
        });
        uploaded++;
      }

      // Step 4: All uploads done -- NOW start processing
      setProgress("Starting processing pipeline...");
      await apiPost(`/jobs/${jobId}/upload-url`, {
        filenames: files.map((f) => f.name),
        start_processing: true,
      });

      // Step 5: Add uploaded documents to selected collections
      if (selectedCollectionIds.size > 0) {
        setProgress("Adding documents to collections...");
        const documents = files.map((f) => ({
          s3_key: `${jobRes.input_s3_prefix}/${f.name}`,
          filename: f.name,
          job_id: jobId,
          source_type: "input",
          size: f.size,
        }));

        const collectionPromises = Array.from(selectedCollectionIds).map((colId) =>
          apiPost(`/collections/${colId}/documents`, {
            action: "add",
            documents,
          }).catch(() => {
            // Best-effort: don't block job creation if collection add fails
          })
        );
        await Promise.all(collectionPromises);
      }

      setProgress("Done! Redirecting...");
      router.push(`/jobs/${jobId}`);
    } catch (err) {
      setProgress(`Error: ${err instanceof Error ? err.message : String(err)}`);
      setUploading(false);
    }
  };

  return (
    <div style={{ maxWidth: "48rem", margin: "0 auto", padding: "2rem" }}>
      {/* Navigation */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "2rem" }}>
        <Link
          href="/dashboard"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.375rem",
            color: "var(--muted)",
            textDecoration: "none",
            fontSize: "0.875rem",
          }}
        >
          &larr; Dashboard
        </Link>
      </div>

      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.5rem" }}>
        Create New Job
      </h1>
      <p style={{ color: "var(--muted)", fontSize: "0.875rem", marginBottom: "2rem" }}>
        Configure your processing job, add optional metadata, and upload documents.
      </p>

      <form onSubmit={handleSubmit}>
        {/* ── Processing Task ──────────────────────────────────────── */}
        <section style={{ marginBottom: "2rem" }}>
          <label style={{ display: "block", fontWeight: 600, fontSize: "0.875rem", marginBottom: "0.5rem" }}>
            Processing Task
          </label>
          <div style={{ display: "grid", gap: "0.5rem" }}>
            {TASK_TYPES.map((t) => (
              <label
                key={t.value}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "0.75rem",
                  padding: "0.75rem 1rem",
                  border: `2px solid ${taskType === t.value ? "var(--primary)" : "var(--border)"}`,
                  borderRadius: "0.5rem",
                  cursor: uploading ? "not-allowed" : "pointer",
                  backgroundColor: taskType === t.value ? "#eff6ff" : "white",
                  transition: "border-color 0.15s, background-color 0.15s",
                }}
              >
                <input
                  type="radio"
                  name="taskType"
                  value={t.value}
                  checked={taskType === t.value}
                  onChange={(e) => setTaskType(e.target.value)}
                  disabled={uploading}
                  style={{ marginTop: "0.125rem", accentColor: "var(--primary)" }}
                />
                <div>
                  <div style={{ fontWeight: 500, fontSize: "0.875rem" }}>{t.label}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "0.125rem" }}>
                    {t.desc}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </section>

        {/* ── OCR Engine ───────────────────────────────────────────── */}
        {(taskType === "full_pipeline" || taskType === "document_extraction") && (
          <section style={{ marginBottom: "2rem" }}>
            <label style={{ display: "block", fontWeight: 600, fontSize: "0.875rem", marginBottom: "0.5rem" }}>
              OCR Engine
            </label>
            <div style={{ display: "grid", gap: "0.5rem" }}>
              {OCR_ENGINES.map((engine) => (
                <label
                  key={engine.value}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "0.75rem",
                    padding: "0.75rem 1rem",
                    border: `2px solid ${ocrEngine === engine.value ? "var(--primary)" : "var(--border)"}`,
                    borderRadius: "0.5rem",
                    cursor: uploading ? "not-allowed" : "pointer",
                    backgroundColor: ocrEngine === engine.value ? "#eff6ff" : "white",
                    transition: "border-color 0.15s, background-color 0.15s",
                  }}
                >
                  <input
                    type="radio"
                    name="ocrEngine"
                    value={engine.value}
                    checked={ocrEngine === engine.value}
                    onChange={(e) => setOcrEngine(e.target.value)}
                    disabled={uploading}
                    style={{ marginTop: "0.125rem", accentColor: "var(--primary)" }}
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span style={{ fontWeight: 500, fontSize: "0.875rem" }}>{engine.label}</span>
                      {engine.tag && (
                        <span
                          style={{
                            fontSize: "0.65rem",
                            fontWeight: 600,
                            padding: "0.0625rem 0.375rem",
                            borderRadius: "0.25rem",
                            backgroundColor: ocrEngine === engine.value ? "#dbeafe" : "#f3f4f6",
                            color: ocrEngine === engine.value ? "var(--primary)" : "var(--muted)",
                          }}
                        >
                          {engine.tag}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "0.125rem" }}>
                      {engine.desc}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </section>
        )}

        {/* ── Custom ID ────────────────────────────────────────────── */}
        <section style={{ marginBottom: "2rem" }}>
          <label
            htmlFor="customId"
            style={{ display: "block", fontWeight: 600, fontSize: "0.875rem", marginBottom: "0.25rem" }}
          >
            Identifier
            <span style={{ fontWeight: 400, color: "var(--muted)", marginLeft: "0.375rem" }}>
              (optional)
            </span>
          </label>
          <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
            A human-readable ID for this job, such as an accession number or collection name.
          </p>
          <input
            id="customId"
            type="text"
            value={customId}
            onChange={(e) => setCustomId(e.target.value)}
            disabled={uploading}
            placeholder="e.g. NUL-2026-0042"
            style={{
              width: "100%",
              padding: "0.5rem 0.75rem",
              border: "1px solid var(--border)",
              borderRadius: "0.375rem",
              fontSize: "0.875rem",
              backgroundColor: "white",
              outline: "none",
              boxSizing: "border-box",
            }}
          />
        </section>

        {/* ── Metadata ─────────────────────────────────────────────── */}
        <section style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
            <label style={{ fontWeight: 600, fontSize: "0.875rem" }}>
              Metadata
              <span style={{ fontWeight: 400, color: "var(--muted)", marginLeft: "0.375rem" }}>
                (optional)
              </span>
            </label>
            <button
              type="button"
              onClick={addMetadataEntry}
              disabled={uploading}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.25rem",
                padding: "0.25rem 0.625rem",
                fontSize: "0.75rem",
                fontWeight: 500,
                color: "var(--primary)",
                backgroundColor: "transparent",
                border: "1px solid var(--primary)",
                borderRadius: "0.375rem",
                cursor: uploading ? "not-allowed" : "pointer",
              }}
            >
              + Add Field
            </button>
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
            Add key-value pairs to describe this job (e.g. collection, source, date range).
          </p>

          {metadataEntries.length === 0 ? (
            <div
              style={{
                padding: "1.25rem",
                border: "1px dashed var(--border)",
                borderRadius: "0.5rem",
                textAlign: "center",
                color: "var(--muted)",
                fontSize: "0.8rem",
              }}
            >
              No metadata fields added yet. Click &quot;+ Add Field&quot; to add one.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {metadataEntries.map((entry) => (
                <div
                  key={entry.id}
                  style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}
                >
                  <input
                    type="text"
                    value={entry.key}
                    onChange={(e) => updateMetadataEntry(entry.id, "key", e.target.value)}
                    placeholder="Key"
                    disabled={uploading}
                    style={{
                      flex: "0 0 35%",
                      padding: "0.5rem 0.75rem",
                      border: "1px solid var(--border)",
                      borderRadius: "0.375rem",
                      fontSize: "0.8rem",
                      fontWeight: 500,
                      backgroundColor: "white",
                      outline: "none",
                    }}
                  />
                  <input
                    type="text"
                    value={entry.value}
                    onChange={(e) => updateMetadataEntry(entry.id, "value", e.target.value)}
                    placeholder="Value"
                    disabled={uploading}
                    style={{
                      flex: 1,
                      padding: "0.5rem 0.75rem",
                      border: "1px solid var(--border)",
                      borderRadius: "0.375rem",
                      fontSize: "0.8rem",
                      backgroundColor: "white",
                      outline: "none",
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => removeMetadataEntry(entry.id)}
                    disabled={uploading}
                    style={{
                      flex: "0 0 auto",
                      padding: "0.375rem 0.5rem",
                      fontSize: "0.875rem",
                      color: "var(--muted)",
                      backgroundColor: "transparent",
                      border: "1px solid var(--border)",
                      borderRadius: "0.375rem",
                      cursor: uploading ? "not-allowed" : "pointer",
                      lineHeight: 1,
                    }}
                    title="Remove field"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ── Collections ──────────────────────────────────────────── */}
        <section style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
            <label style={{ fontWeight: 600, fontSize: "0.875rem" }}>
              Collections
              <span style={{ fontWeight: 400, color: "var(--muted)", marginLeft: "0.375rem" }}>
                (optional)
              </span>
            </label>
            <button
              type="button"
              onClick={() => setShowCreateCollection(true)}
              disabled={uploading}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.25rem",
                padding: "0.25rem 0.625rem",
                fontSize: "0.75rem",
                fontWeight: 500,
                color: "var(--primary)",
                backgroundColor: "transparent",
                border: "1px solid var(--primary)",
                borderRadius: "0.375rem",
                cursor: uploading ? "not-allowed" : "pointer",
              }}
            >
              + New Collection
            </button>
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
            Add this job&apos;s documents to one or more collections.
          </p>

          {/* Inline create collection form */}
          {showCreateCollection && (
            <div
              style={{
                padding: "0.75rem",
                border: "2px solid var(--primary)",
                borderRadius: "0.5rem",
                backgroundColor: "#eff6ff",
                marginBottom: "0.5rem",
              }}
            >
              <div style={{ fontWeight: 500, fontSize: "0.8rem", marginBottom: "0.5rem" }}>
                Create new collection
              </div>
              <input
                type="text"
                value={newCollectionName}
                onChange={(e) => setNewCollectionName(e.target.value)}
                placeholder="Collection name"
                disabled={creatingCollection}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid var(--border)",
                  borderRadius: "0.375rem",
                  fontSize: "0.8rem",
                  backgroundColor: "white",
                  outline: "none",
                  boxSizing: "border-box",
                  marginBottom: "0.375rem",
                }}
              />
              <input
                type="text"
                value={newCollectionDescription}
                onChange={(e) => setNewCollectionDescription(e.target.value)}
                placeholder="Description (optional)"
                disabled={creatingCollection}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid var(--border)",
                  borderRadius: "0.375rem",
                  fontSize: "0.8rem",
                  backgroundColor: "white",
                  outline: "none",
                  boxSizing: "border-box",
                  marginBottom: "0.5rem",
                }}
              />
              <div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>
                <button
                  type="button"
                  onClick={() => {
                    setShowCreateCollection(false);
                    setNewCollectionName("");
                    setNewCollectionDescription("");
                  }}
                  disabled={creatingCollection}
                  style={{
                    padding: "0.375rem 0.75rem",
                    fontSize: "0.75rem",
                    backgroundColor: "transparent",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    cursor: creatingCollection ? "not-allowed" : "pointer",
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleCreateCollection}
                  disabled={creatingCollection || !newCollectionName.trim()}
                  style={{
                    padding: "0.375rem 0.75rem",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "white",
                    backgroundColor: creatingCollection || !newCollectionName.trim() ? "#9ca3af" : "var(--primary)",
                    border: "none",
                    borderRadius: "0.375rem",
                    cursor: creatingCollection || !newCollectionName.trim() ? "not-allowed" : "pointer",
                  }}
                >
                  {creatingCollection ? "Creating..." : "Create"}
                </button>
              </div>
            </div>
          )}

          {/* Collection list */}
          {loadingCollections ? (
            <div
              style={{
                padding: "1.25rem",
                border: "1px dashed var(--border)",
                borderRadius: "0.5rem",
                textAlign: "center",
                color: "var(--muted)",
                fontSize: "0.8rem",
              }}
            >
              Loading collections...
            </div>
          ) : availableCollections.length === 0 && !showCreateCollection ? (
            <div
              style={{
                padding: "1.25rem",
                border: "1px dashed var(--border)",
                borderRadius: "0.5rem",
                textAlign: "center",
                color: "var(--muted)",
                fontSize: "0.8rem",
              }}
            >
              No collections yet. Click &quot;+ New Collection&quot; to create one.
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.375rem",
                maxHeight: "14rem",
                overflowY: "auto",
              }}
            >
              {availableCollections.map((col) => {
                const selected = selectedCollectionIds.has(col.collection_id);
                return (
                  <label
                    key={col.collection_id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.75rem",
                      padding: "0.625rem 0.75rem",
                      border: `2px solid ${selected ? "var(--primary)" : "var(--border)"}`,
                      borderRadius: "0.5rem",
                      cursor: uploading ? "not-allowed" : "pointer",
                      backgroundColor: selected ? "#eff6ff" : "white",
                      transition: "border-color 0.15s, background-color 0.15s",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleCollection(col.collection_id)}
                      disabled={uploading}
                      style={{ accentColor: "var(--primary)" }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 500, fontSize: "0.85rem" }}>{col.name}</div>
                    </div>
                    <span
                      style={{
                        fontSize: "0.7rem",
                        color: "var(--muted)",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {col.document_count} doc{col.document_count !== 1 ? "s" : ""}
                    </span>
                  </label>
                );
              })}
            </div>
          )}
        </section>

        {/* ── File Upload ──────────────────────────────────────────── */}
        <section style={{ marginBottom: "2rem" }}>
          <label style={{ display: "block", fontWeight: 600, fontSize: "0.875rem", marginBottom: "0.5rem" }}>
            Documents
          </label>

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            style={{
              position: "relative",
              padding: "2rem",
              border: `2px dashed ${dragActive ? "var(--primary)" : "var(--border)"}`,
              borderRadius: "0.5rem",
              textAlign: "center",
              backgroundColor: dragActive ? "#eff6ff" : "white",
              transition: "border-color 0.15s, background-color 0.15s",
              cursor: "pointer",
            }}
            onClick={() => document.getElementById("fileInput")?.click()}
          >
            <input
              id="fileInput"
              type="file"
              multiple
              onChange={handleFileChange}
              disabled={uploading}
              accept=".pdf,.jp2,.jpg,.jpeg,.png,.tiff,.tif,.xml"
              style={{ display: "none" }}
            />
            <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem", opacity: 0.3 }}>
              &#8593;
            </div>
            <p style={{ fontWeight: 500, fontSize: "0.875rem", marginBottom: "0.25rem" }}>
              Drop files here or click to browse
            </p>
            <p style={{ fontSize: "0.75rem", color: "var(--muted)" }}>
              PDF, JP2, JPEG, PNG, TIFF, XML
            </p>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div style={{ marginTop: "0.75rem" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
                {files.length} file{files.length !== 1 ? "s" : ""} selected ({formatSize(files.reduce((s, f) => s + f.size, 0))} total)
              </div>
              <div
                style={{
                  maxHeight: "12rem",
                  overflowY: "auto",
                  border: "1px solid var(--border)",
                  borderRadius: "0.375rem",
                }}
              >
                {files.map((file, i) => (
                  <div
                    key={`${file.name}-${i}`}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "0.375rem 0.75rem",
                      borderBottom: i < files.length - 1 ? "1px solid var(--border)" : "none",
                      fontSize: "0.8rem",
                    }}
                  >
                    <span style={{ fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "70%" }}>
                      {file.name}
                    </span>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <span style={{ color: "var(--muted)", fontSize: "0.75rem" }}>
                        {formatSize(file.size)}
                      </span>
                      {!uploading && (
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                          style={{
                            color: "var(--muted)",
                            backgroundColor: "transparent",
                            border: "none",
                            cursor: "pointer",
                            fontSize: "1rem",
                            lineHeight: 1,
                            padding: "0.125rem",
                          }}
                        >
                          &times;
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* ── Submit ────────────────────────────────────────────────── */}
        <button
          type="submit"
          disabled={uploading || files.length === 0}
          style={{
            width: "100%",
            padding: "0.75rem",
            backgroundColor: uploading || files.length === 0 ? "#9ca3af" : "var(--primary)",
            color: "white",
            border: "none",
            borderRadius: "0.5rem",
            fontWeight: 600,
            fontSize: "0.875rem",
            cursor: uploading || files.length === 0 ? "not-allowed" : "pointer",
            transition: "background-color 0.15s",
          }}
        >
          {uploading ? "Processing..." : "Create Job & Upload"}
        </button>

        {progress && (
          <div
            style={{
              marginTop: "1rem",
              padding: "0.75rem 1rem",
              borderRadius: "0.375rem",
              fontSize: "0.875rem",
              backgroundColor: progress.startsWith("Error") ? "#fef2f2" : "#f0fdf4",
              color: progress.startsWith("Error") ? "var(--danger)" : "var(--success)",
              border: `1px solid ${progress.startsWith("Error") ? "#fecaca" : "#bbf7d0"}`,
            }}
          >
            {progress}
          </div>
        )}
      </form>
    </div>
  );
}
