"""Analysis API handlers.

CRUD for analyses + the run engine that computes NER, geocode,
word frequencies, text similarity, and summary statistics.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone

import boto3
from loguru import logger

from impulse.db.client import get_collection as get_mongo_collection
from impulse.db.models import Analysis

# Stop words for word frequency analysis
_STOP_WORDS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would shall should may might can could not no nor of in on at to for "
    "with from by as into through during before after above below between out off "
    "over under again further then once here there when where why how all each "
    "every both few more most other some such only own same so than too very it "
    "its he she they them their his her this that these those i me my we us our "
    "you your what which who whom whose if while also about up just because "
    "am been being did doing having he'd he'll he's i'd i'll i'm i've let's "
    "she'd she'll she's that's they'd they'll they're they've we'd we'll we're "
    "we've what's when's where's who's why's would been being".split()
)


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }


def _col():
    return get_mongo_collection("analyses")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── POST /analyses ───────────────────────────────────────────────────────


def create_analysis(body: dict, user_id: str) -> dict:
    name = body.get("name", "").strip()
    if not name:
        return _response(400, {"error": "Analysis name is required"})

    analysis = Analysis(
        analysis_id=str(uuid.uuid4()),
        user_id=user_id,
        name=name,
        description=body.get("description", "").strip(),
        sources=body.get("sources", []),
    )

    _col().insert_one(analysis.to_dict())
    logger.info(f"Created analysis {analysis.analysis_id}")

    return _response(
        201,
        {
            "analysis_id": analysis.analysis_id,
            "name": analysis.name,
        },
    )


# ── GET /analyses ────────────────────────────────────────────────────────


def list_analyses(user_id: str) -> dict:
    cursor = (
        _col()
        .find(
            {"user_id": user_id},
            {
                "_id": 0,
                "analysis_id": 1,
                "name": 1,
                "description": 1,
                "status": 1,
                "sources": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
        .sort("updated_at", -1)
    )

    analyses = []
    for a in cursor:
        a["source_count"] = len(a.get("sources", []))
        analyses.append(a)

    return _response(200, {"analyses": analyses, "count": len(analyses)})


# ── GET /analyses/{analysisId} ───────────────────────────────────────────


def get_analysis(analysis_id: str, user_id: str) -> dict:
    a = _col().find_one(
        {"analysis_id": analysis_id, "user_id": user_id},
        {"_id": 0},
    )
    if not a:
        return _response(404, {"error": "Analysis not found"})

    return _response(200, {"analysis": a})


# ── DELETE /analyses/{analysisId} ────────────────────────────────────────


def delete_analysis(analysis_id: str, user_id: str, claims: dict | None = None) -> dict:
    from impulse.api.auth import require_admin

    denied = require_admin(claims or {})
    if denied:
        return denied

    result = _col().delete_one(
        {"analysis_id": analysis_id, "user_id": user_id},
    )
    if result.deleted_count == 0:
        return _response(404, {"error": "Analysis not found"})

    return _response(200, {"message": "Analysis deleted"})


# ── PUT /analyses/{analysisId}/sources ───────────────────────────────────


def update_sources(analysis_id: str, body: dict, user_id: str) -> dict:
    a = _col().find_one(
        {"analysis_id": analysis_id, "user_id": user_id},
        {"_id": 0, "analysis_id": 1},
    )
    if not a:
        return _response(404, {"error": "Analysis not found"})

    sources = body.get("sources", [])
    _col().update_one(
        {"analysis_id": analysis_id},
        {"$set": {"sources": sources, "updated_at": _now()}},
    )

    return _response(200, {"message": "Sources updated"})


# ── POST /analyses/{analysisId}/run ──────────────────────────────────────


def run_analysis(analysis_id: str, user_id: str) -> dict:
    """Run all analysis computations on the selected sources."""
    a = _col().find_one(
        {"analysis_id": analysis_id, "user_id": user_id},
        {"_id": 0},
    )
    if not a:
        return _response(404, {"error": "Analysis not found"})

    # Mark as running
    _col().update_one(
        {"analysis_id": analysis_id},
        {"$set": {"status": "RUNNING", "updated_at": _now()}},
    )

    try:
        sources = a.get("sources", [])

        # Step 1: Gather all extracted text from sources
        docs = _gather_documents(sources, user_id)
        logger.info(f"Analysis {analysis_id}: gathered {len(docs)} documents")

        if not docs:
            _col().update_one(
                {"analysis_id": analysis_id},
                {
                    "$set": {
                        "status": "COMPLETED",
                        "summary_stats": {"total_docs": 0, "total_chars": 0},
                        "updated_at": _now(),
                    }
                },
            )
            return _response(200, {"message": "No documents found in sources"})

        # Step 2: NER extraction
        entities, entity_edges = _compute_ner(docs)

        # Step 3: Geocode locations
        locations = _geocode_locations(entities)

        # Step 4: Word frequencies
        word_frequencies = _compute_word_frequencies(docs)

        # Step 5: Text similarity / clustering
        doc_coordinates = _compute_similarity(docs)

        # Step 6: Timeline events
        timeline_events = _compute_timeline(sources, user_id)

        # Step 7: Summary stats
        summary_stats = _compute_summary(docs, entities, sources, user_id)

        # Save all results
        _col().update_one(
            {"analysis_id": analysis_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "entities": entities[:200],  # Cap for document size
                    "entity_edges": entity_edges[:500],
                    "locations": locations[:100],
                    "word_frequencies": word_frequencies[:100],
                    "doc_coordinates": doc_coordinates,
                    "timeline_events": timeline_events,
                    "summary_stats": summary_stats,
                    "updated_at": _now(),
                }
            },
        )

        logger.info(
            f"Analysis {analysis_id} completed: {len(entities)} entities, {len(locations)} locations"
        )
        return _response(200, {"message": "Analysis completed", "status": "COMPLETED"})

    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        _col().update_one(
            {"analysis_id": analysis_id},
            {"$set": {"status": "FAILED", "updated_at": _now()}},
        )
        return _response(500, {"error": str(e)})


# ── Internal computation functions ───────────────────────────────────────


def _gather_documents(sources: list[dict], user_id: str) -> list[dict]:
    """Collect all documents with extracted text from the given sources.

    Returns list of {doc_key, filename, text, job_id}.
    """
    results_col = get_mongo_collection("results")
    jobs_col = get_mongo_collection("jobs")
    collections_col = get_mongo_collection("collections")

    doc_map: dict[str, dict] = {}

    for source in sources:
        src_type = source.get("type", "")
        src_id = source.get("id", "")

        if src_type == "job":
            # Get all results for this job
            for r in results_col.find(
                {"job_id": src_id},
                {"_id": 0, "document_key": 1, "extracted_text": 1, "job_id": 1},
            ):
                key = r.get("document_key", "")
                if key and r.get("extracted_text"):
                    doc_map[key] = {
                        "doc_key": key,
                        "filename": key.split("/")[-1],
                        "text": r["extracted_text"],
                        "job_id": r.get("job_id", src_id),
                    }

        elif src_type == "collection":
            col = collections_col.find_one(
                {"collection_id": src_id, "user_id": user_id},
                {"_id": 0, "documents": 1},
            )
            if col:
                for doc in col.get("documents", []):
                    s3_key = doc.get("s3_key", "")
                    job_id = doc.get("job_id", "")
                    # Look up extracted text
                    r = results_col.find_one(
                        {"document_key": s3_key},
                        {"_id": 0, "extracted_text": 1},
                    )
                    if r and r.get("extracted_text"):
                        doc_map[s3_key] = {
                            "doc_key": s3_key,
                            "filename": doc.get("filename", s3_key.split("/")[-1]),
                            "text": r["extracted_text"],
                            "job_id": job_id,
                        }

    return list(doc_map.values())


def _compute_ner(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run NER on all documents using simple regex-based entity extraction.

    For production, this would use the BERT NER model via Lambda, but for
    the analysis engine running inside the API Lambda (limited to 512MB),
    we use a lightweight pattern-based approach + the spaCy entities
    already stored in the metadata collection.
    """
    entity_counter: Counter = Counter()
    entity_types: dict[str, str] = {}
    entity_docs: dict[str, set] = defaultdict(set)

    # First: check if metadata collection has pre-computed entities
    metadata_col = get_mongo_collection("metadata")
    for doc in docs:
        job_id = doc.get("job_id", "")
        # Check metadata for key_people and main_place
        meta = metadata_col.find_one(
            {"accession_number": {"$in": [job_id, ""]}},
            {"_id": 0, "key_people": 1, "main_place": 1},
        )
        if meta:
            for person in meta.get("key_people", []) or []:
                if person and len(person) > 1:
                    entity_counter[person] += 1
                    entity_types[person] = "PER"
                    entity_docs[person].add(doc["doc_key"])
            place = meta.get("main_place")
            if place and len(place) > 1:
                entity_counter[place] += 1
                entity_types[place] = "LOC"
                entity_docs[place].add(doc["doc_key"])

    # Fallback: regex-based extraction for capitalized multi-word phrases
    # This catches proper nouns that spaCy/metadata might have missed
    cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    for doc in docs:
        text = doc["text"]
        for match in cap_pattern.finditer(text):
            phrase = match.group(1)
            # Skip very common phrases
            if phrase.lower() in _STOP_WORDS or len(phrase) < 4:
                continue
            entity_counter[phrase] += 1
            if phrase not in entity_types:
                # Heuristic: if it looks like a place (ends with common suffixes)
                lower = phrase.lower()
                if any(
                    lower.endswith(s)
                    for s in (
                        "city",
                        "county",
                        "state",
                        "park",
                        "lake",
                        "river",
                        "mountain",
                        "valley",
                        "island",
                        "creek",
                        "springs",
                    )
                ):
                    entity_types[phrase] = "LOC"
                elif any(
                    lower.endswith(s)
                    for s in (
                        "university",
                        "college",
                        "institute",
                        "corporation",
                        "company",
                        "department",
                        "bureau",
                        "commission",
                        "agency",
                        "association",
                        "foundation",
                        "library",
                        "museum",
                    )
                ):
                    entity_types[phrase] = "ORG"
                else:
                    entity_types[phrase] = "PER"
            entity_docs[phrase].add(doc["doc_key"])

    # Build entity list (top 200 by count)
    entities = []
    for text, count in entity_counter.most_common(200):
        entities.append(
            {
                "text": text,
                "type": entity_types.get(text, "MISC"),
                "count": count,
                "documents": list(entity_docs[text])[:20],
            }
        )

    # Build co-occurrence edges
    edges = _build_cooccurrence_edges(entity_docs, entity_counter)

    return entities, edges


def _build_cooccurrence_edges(
    entity_docs: dict[str, set],
    entity_counter: Counter,
) -> list[dict]:
    """Build edges between entities that co-occur in the same documents."""
    # Only consider top entities to keep graph manageable
    top_entities = [e for e, _ in entity_counter.most_common(80)]
    edges: list[dict] = []
    seen: set[tuple] = set()

    for i, e1 in enumerate(top_entities):
        for e2 in top_entities[i + 1 :]:
            shared = entity_docs[e1] & entity_docs[e2]
            if shared:
                pair = tuple(sorted([e1, e2]))
                if pair not in seen:
                    seen.add(pair)
                    edges.append(
                        {
                            "source": e1,
                            "target": e2,
                            "weight": len(shared),
                            "documents": list(shared)[:10],
                        }
                    )

    # Sort by weight descending, cap at 500
    edges.sort(key=lambda e: e["weight"], reverse=True)
    return edges[:500]


def _geocode_locations(entities: list[dict]) -> list[dict]:
    """Geocode LOC entities using Nominatim (with rate limiting)."""
    import urllib.request
    import urllib.parse

    locations: list[dict] = []
    seen: set[str] = set()

    loc_entities = [e for e in entities if e["type"] == "LOC"][:30]  # Cap at 30

    for entity in loc_entities:
        name = entity["text"]
        if name.lower() in seen:
            continue
        seen.add(name.lower())

        try:
            encoded = urllib.parse.quote(name)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Impulse/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data:
                    locations.append(
                        {
                            "name": name,
                            "lat": float(data[0]["lat"]),
                            "lon": float(data[0]["lon"]),
                            "display_name": data[0].get("display_name", name),
                            "count": entity["count"],
                            "documents": entity["documents"],
                        }
                    )
            time.sleep(1.1)  # Nominatim rate limit
        except Exception as e:
            logger.warning(f"Geocode failed for {name}: {e}")

    return locations


def _compute_word_frequencies(docs: list[dict]) -> list[dict]:
    """Compute word frequencies across all documents."""
    counter: Counter = Counter()
    word_pattern = re.compile(r"[a-zA-Z]{3,}")

    for doc in docs:
        words = word_pattern.findall(doc["text"].lower())
        for word in words:
            if word not in _STOP_WORDS and len(word) > 2:
                counter[word] += 1

    return [{"word": w, "count": c} for w, c in counter.most_common(100)]


def _compute_similarity(docs: list[dict]) -> list[dict]:
    """Compute 2D document coordinates using TF-IDF + simple dimensionality reduction.

    Uses a pure-Python implementation (no numpy dependency in Lambda).
    """
    if len(docs) < 2:
        return [
            {"doc_key": d["doc_key"], "filename": d["filename"], "x": 0, "y": 0}
            for d in docs
        ]

    word_pattern = re.compile(r"[a-zA-Z]{3,}")

    # Build vocabulary from top-N words
    global_counter: Counter = Counter()
    doc_words: list[list[str]] = []
    for doc in docs:
        words = [
            w for w in word_pattern.findall(doc["text"].lower()) if w not in _STOP_WORDS
        ]
        doc_words.append(words)
        global_counter.update(set(words))  # Document frequency

    # Top 200 terms by document frequency
    vocab = [w for w, _ in global_counter.most_common(200)]
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    if not vocab:
        return [
            {"doc_key": d["doc_key"], "filename": d["filename"], "x": 0, "y": 0}
            for d in docs
        ]

    n_docs = len(docs)
    n_vocab = len(vocab)

    # Build TF-IDF vectors
    vectors: list[list[float]] = []
    for words in doc_words:
        tf: Counter = Counter(words)
        vec = [0.0] * n_vocab
        for word, count in tf.items():
            if word in vocab_idx:
                # TF-IDF: tf * log(N / df)
                df = global_counter[word]
                idf = math.log(n_docs / max(df, 1))
                vec[vocab_idx[word]] = count * idf
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        vectors.append(vec)

    # Simple 2D projection: use first two "principal components" via power iteration
    # This is a very simplified PCA -- sufficient for visualization
    coords = _simple_2d_projection(vectors)

    return [
        {
            "doc_key": docs[i]["doc_key"],
            "filename": docs[i]["filename"],
            "x": round(coords[i][0], 4),
            "y": round(coords[i][1], 4),
        }
        for i in range(len(docs))
    ]


def _simple_2d_projection(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Project high-dimensional vectors to 2D using simplified SVD."""
    n = len(vectors)
    if n == 0:
        return []
    d = len(vectors[0])

    # Compute mean
    mean = [0.0] * d
    for vec in vectors:
        for j in range(d):
            mean[j] += vec[j]
    mean = [m / n for m in mean]

    # Center
    centered = [[vec[j] - mean[j] for j in range(d)] for vec in vectors]

    # Power iteration to find top 2 directions
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def mat_vec(mat: list[list[float]], v: list[float]) -> list[float]:
        return [dot(row, v) for row in mat]

    def vec_norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    import random

    random.seed(42)

    results: list[list[float]] = []
    deflated = [row[:] for row in centered]

    for _ in range(2):
        # Random init
        v = [random.gauss(0, 1) for _ in range(d)]
        norm = vec_norm(v)
        v = [x / max(norm, 1e-10) for x in v]

        # Power iteration (X^T X v)
        for _ in range(50):
            # Compute X^T (X v)
            xv = mat_vec(deflated, v)  # n-dim
            xtxv = [0.0] * d
            for i_row in range(n):
                for j_col in range(d):
                    xtxv[j_col] += deflated[i_row][j_col] * xv[i_row]
            norm = vec_norm(xtxv)
            if norm < 1e-10:
                break
            v = [x / norm for x in xtxv]

        # Project
        proj = mat_vec(deflated, v)
        results.append(proj)

        # Deflate
        for i_row in range(n):
            for j_col in range(d):
                deflated[i_row][j_col] -= proj[i_row] * v[j_col]

    if len(results) < 2:
        results.append([0.0] * n)

    return list(zip(results[0], results[1]))


def _compute_timeline(sources: list[dict], user_id: str) -> list[dict]:
    """Build timeline events from source jobs."""
    jobs_col = get_mongo_collection("jobs")
    events: list[dict] = []

    job_ids = [s["id"] for s in sources if s.get("type") == "job"]
    if not job_ids:
        # Get jobs from all sources
        for s in sources:
            if s.get("type") == "collection":
                col = get_mongo_collection("collections").find_one(
                    {"collection_id": s["id"]},
                    {"_id": 0, "documents": 1},
                )
                if col:
                    job_ids.extend(
                        {
                            d.get("job_id")
                            for d in col.get("documents", [])
                            if d.get("job_id")
                        }
                    )

    for job_id in set(job_ids):
        job = jobs_col.find_one(
            {"job_id": job_id},
            {
                "_id": 0,
                "job_id": 1,
                "custom_id": 1,
                "status": 1,
                "task_type": 1,
                "ocr_engine": 1,
                "total_documents": 1,
                "processed_documents": 1,
                "failed_documents": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
        if job:
            events.append(job)

    events.sort(key=lambda e: e.get("created_at", ""))
    return events


def _compute_summary(
    docs: list[dict],
    entities: list[dict],
    sources: list[dict],
    user_id: str,
) -> dict:
    """Compute aggregate summary statistics."""
    jobs_col = get_mongo_collection("jobs")

    total_chars = sum(len(d["text"]) for d in docs)
    total_words = sum(len(d["text"].split()) for d in docs)

    # Entity type breakdown
    type_counts: Counter = Counter()
    for e in entities:
        type_counts[e["type"]] += e["count"]

    # OCR engine breakdown
    ocr_engines: Counter = Counter()
    job_ids = {d["job_id"] for d in docs if d.get("job_id")}
    for job_id in job_ids:
        job = jobs_col.find_one(
            {"job_id": job_id},
            {"_id": 0, "ocr_engine": 1},
        )
        if job:
            ocr_engines[job.get("ocr_engine", "unknown")] += 1

    return {
        "total_docs": len(docs),
        "total_chars": total_chars,
        "total_words": total_words,
        "unique_entities": len(entities),
        "entity_type_counts": dict(type_counts),
        "ocr_engine_counts": dict(ocr_engines),
        "top_entities": [
            {"text": e["text"], "type": e["type"], "count": e["count"]}
            for e in entities[:10]
        ],
        "source_count": len(sources),
    }
