# Impulse

![Impulse Logo](logo.png)

Serverless document processing pipeline for large-scale digitization, OCR, and metadata extraction. Built on AWS with Step Functions, ECS Fargate, Lambda, and MongoDB Atlas.

## Architecture

```
Frontend (Next.js / Amplify)
    |
API Gateway + Cognito Auth
    |
Lambda (Job CRUD)  -->  MongoDB Atlas (metadata only, no binaries)
    |
Step Functions (orchestration)
    |
    +-- ECS Fargate: Image processing (OpenCV, binarization, denoising)
    +-- ECS Fargate: Document extraction (Marker PDF OCR)
    +-- Lambda: METS conversion, metadata extraction, NER, geocoding, summaries
    +-- Amazon Bedrock: LLM-powered metadata + summarisation
    |
S3 (all document I/O)
```

## Project Structure

```
Impulse/
├── src/impulse/
│   ├── api/            # Lambda API handlers (job CRUD, presigned uploads)
│   ├── db/             # MongoDB models and client
│   ├── handlers/       # ECS and Lambda entry points
│   ├── processing/     # Core processing logic (stateless, no framework deps)
│   ├── config.py       # Centralised configuration from env vars
│   └── utils.py        # S3 helpers, file detection, shared utilities
├── infra/              # AWS CDK infrastructure-as-code
│   ├── stacks/         # Storage, network, auth, API, processing stacks
│   └── app.py          # CDK app entry point
├── frontend/           # Next.js web application
├── docker/             # Dockerfiles for ECS Fargate tasks
├── tests/              # Unit tests
└── scripts/            # Dev scripts
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS account with CDK bootstrapped
- MongoDB Atlas cluster
- Node.js 20+ (for frontend)

### Local Development

```bash
# Install Python dependencies
uv sync

# Copy and configure environment
cp .env_sample .env
# Edit .env with your MongoDB URI, etc.

# Run tests
uv run pytest tests/ -v

# Run a specific processing module locally
MONGODB_URI="mongodb://..." IMPULSE_BUCKET="my-bucket" \
  uv run python -c "from impulse.processing.images import process_image; ..."
```

### Deploy Infrastructure

```bash
cd infra
uv venv && uv pip install -r pyproject.toml
cdk deploy --all
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Fill in Cognito and API Gateway values from CDK outputs
npm run dev
```

## Processing Tasks

| Task | Compute | Description |
|------|---------|-------------|
| Image Transform | ECS Fargate | Grayscale, Otsu binarization, NLM denoising, JP2 encoding |
| Document Extraction | ECS Fargate | Marker PDF for OCR, structured JSON output |
| METS Conversion | Lambda | METS XML to HathiTrust YAML manifests |
| Metadata Extraction | Lambda + Bedrock | SpaCy NER + Claude for place/people extraction |
| Summarisation | Lambda + Bedrock | Document summaries via Claude |
| NER | Lambda | BERT-based named entity recognition |
| Geocoding | Lambda | OpenStreetMap Nominatim geocoding |

## Credits

- [Marker PDF by DataLab](https://github.com/datalab-to/marker)
- [SpaCy](https://spacy.io/)
- [OpenCV](https://opencv.org/)
