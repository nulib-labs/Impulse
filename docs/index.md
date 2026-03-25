# Impulse Architecture

## Overview

Impulse is a serverless document processing pipeline running on AWS.
Users submit jobs through a web frontend; documents are uploaded directly
to S3 via presigned URLs and processed via Step Functions orchestrating
ECS Fargate tasks and Lambda functions.

## Data Flow

1. User signs in via Cognito, creates a job via the API.
2. Frontend gets presigned S3 upload URLs and uploads files directly to S3.
3. API starts a Step Functions execution.
4. Step Functions validates the job, fans out over documents using Distributed Map.
5. Each document is routed to the appropriate processor:
   - **Image Transform / Document Extraction** -- ECS Fargate (heavy compute)
   - **METS, Metadata, NER, Geocode, Summaries** -- Lambda (lightweight)
6. Results are written to MongoDB Atlas (metadata/text only -- never binary).
7. Processed images/outputs are written back to S3.
8. Job status is updated in MongoDB and visible in the dashboard.

## Key Design Decisions

- **No binary data in MongoDB.** All images and documents live in S3.
  MongoDB stores only metadata, job state, and extracted text.
- **IAM roles for AWS auth.** No hardcoded credentials or named profiles.
  Lambda and ECS tasks use their execution role via the default credential chain.
- **Singleton clients.** S3 and MongoDB clients are created once per process
  and reused across invocations (important for Lambda warm starts).
- **Amazon Bedrock** replaces the previous local Ollama/Gemma3 dependency
  for LLM-powered metadata extraction and summarisation.

## MongoDB Collections

| Collection | Purpose | Binary data? |
|------------|---------|--------------|
| `jobs`     | Job records (status, progress, S3 prefixes) | No |
| `results`  | Per-document extraction results | No |
| `metadata` | Extracted place/people metadata | No |

## Infrastructure (CDK)

The `infra/` directory contains five CDK stacks:

- **StorageStack** -- S3 bucket, ECR repositories
- **NetworkStack** -- VPC, subnets, VPC endpoints (S3, Secrets Manager, ECR)
- **AuthStack** -- Cognito User Pool and app client
- **ProcessingStack** -- ECS cluster, Fargate task defs, Lambda functions, Step Functions state machine, SQS DLQ
- **ApiStack** -- API Gateway REST API, Cognito authorizer, Lambda API handler
