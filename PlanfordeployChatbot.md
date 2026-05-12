# Final Deployment Plan - Vietnamese Legal RAG Chatbot

Target role: **Fullstack AI Integration Intern - FPT Telecom**

Positioning:

```text
Deployable fullstack RAG product
+ async document processing extension
+ optional fine-tuned LLM provider
```

This project should not be presented as only a simple chatbot. The best story is that the core demo is stable and cloud-deployable, while advanced modules are kept as optional extensions.

---

## 1. Final Architecture

### Core Demo

This is the version to deploy, demo, and explain in interviews.

```text
User
  -> Cloud Run Frontend - Streamlit
  -> Cloud Run Backend - FastAPI
  -> Qdrant Cloud - vector database
  -> Gemini API - LLM provider
  -> Cloud SQL - chat history / app state
```

Core services:

| Component | Service | Role |
| --- | --- | --- |
| Frontend | Cloud Run | Streamlit UI for chatbot demo |
| Backend | Cloud Run | FastAPI chat API, RAG orchestration |
| Vector DB | Qdrant Cloud | Store and retrieve legal document embeddings |
| Database | Cloud SQL | Store conversations, users, metadata if needed |
| LLM | Gemini API | Default stable response generation |

Why this is the right default:

- Runs without GPU.
- Easy to deploy and explain.
- Shows fullstack AI integration: UI, API, RAG, vector DB, SQL DB, LLM API, Docker, cloud.
- Avoids making the demo depend on a heavy local fine-tuned model.

### Async Extension

This is the production-style extension for document upload, indexing, and long-running jobs.

```text
Cloud Storage
  -> Cloud Run Backend
  -> Celery task
  -> Memorystore Redis / Valkey-compatible broker
  -> Cloud Run Worker
  -> embedding + chunking + indexing
  -> Qdrant Cloud
```

Async services:

| Component | Service | Role |
| --- | --- | --- |
| Worker | Cloud Run worker | Run Celery jobs for import/indexing |
| Queue/Broker | Memorystore Redis or Valkey-compatible broker | Celery broker/result backend |
| Document Storage | Cloud Storage | Store uploaded legal files and processed datasets |
| Backend | Cloud Run | Create jobs, expose job status API |
| Qdrant Cloud | External managed service | Store final vectors |

Use this explanation:

```text
The main chat path is synchronous and simple for reliability. Document ingestion and large indexing jobs are separated into an async worker path using Celery and Redis-compatible broker, with files stored in Cloud Storage.
```

### Optional Advanced LLM Provider

Keep fine-tuned LLM support, but do not make it the default deploy path.

```text
Vast.ai GPU / local GPU
  -> fine-tuned Vietnamese legal LLM endpoint
  -> OpenAI-compatible API
  -> FastAPI backend via VIETNAMESE_LLM_API_URL
```

CV-safe wording:

```text
Optional support for a PEFT/LoRA fine-tuned Vietnamese legal LLM through an OpenAI-compatible endpoint.
```

Avoid claiming:

```text
Production-ready self-hosted legal LLM platform.
```

---

## 2. What Can Be Reused From Previous Cloud Run Projects

I checked these existing projects:

```text
/Users/nguyen_bao/Documents/PTIT/nam4/Cloud/mini-social-network
/Users/nguyen_bao/Projects/AIproject/Real-Time-Face-Mask-Compliance-Detection-System
```

Useful patterns to reuse:

- Artifact Registry repository creation.
- `gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com`.
- `gcloud builds submit ... --tag ...`.
- `gcloud run deploy ... --image ...`.
- Cloud Run service reads the injected `PORT` environment variable.
- Health check after deploy.
- Separate services/images for microservice-style deployment.
- Cloud Storage for uploaded files instead of local filesystem.
- Broker-based async processing pattern from the microservice project.

Important lessons:

- Cloud Run filesystem is not persistent. Uploaded legal documents must go to Cloud Storage.
- Cloud Run should not depend on local Docker network names like `chatbot-api`, `qdrant-db`, or `mariadb-tiny`.
- Each Cloud Run service must listen on `0.0.0.0:${PORT}`.
- Secrets should be passed through Secret Manager or Cloud Run secrets, not committed to `.env`.
- Build/deploy scripts should use placeholder variables, not hard-coded personal project IDs.

---

## 3. Current Repo Changes Needed Before Real Cloud Run Deploy

These items were required because the repo was originally designed for local Docker Compose. They have now been implemented for the core Cloud Run path while keeping local defaults.

### Backend

Implemented:

```text
backend/Dockerfile and backend/entrypoint.sh read ${PORT:-8080}.
backend/src/app.py supports FORCE_SYNC_CHAT=true for a core demo without Celery.
```

```bash
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
```

Cloud Run injects `PORT`, usually `8080`.

### Frontend

Implemented:

```text
frontend/entrypoint.sh reads ${PORT:-8051}.
frontend/chat_interface_new.py reads API_BASE_URL from environment variables.
frontend supports CHAT_SYNC_REQUEST=true for core Cloud Run demo.
```

Frontend must call the public/private backend Cloud Run URL, not the Docker Compose hostname.

### Qdrant

Implemented:

```text
backend/src/vectorize.py reads QDRANT_URL and QDRANT_API_KEY.
backend/src/configs.py reads QDRANT_COLLECTION.
```

```env
QDRANT_URL=https://<your-qdrant-cloud-cluster>
QDRANT_API_KEY=<secret>
QDRANT_COLLECTION=llm
QDRANT_VECTOR_SIZE=3072  # default for gemini-embedding-001
QDRANT_DISTANCE=COSINE
```

The backend should initialize Qdrant from environment variables.

Important:

```text
The embedding model used for indexing and query-time retrieval must be the same.
If the deployment switches from the custom embedding service to Gemini embeddings,
recreate the Qdrant collection with the matching vector size and re-import data.
```

### Cloud SQL

Implemented:

```env
MYSQL_HOST
MYSQL_PORT
MYSQL_PASSWORD / MYSQL_ROOT_PASSWORD
MYSQL_DATABASE
CLOUD_SQL_CONNECTION_NAME
```

For Cloud Run + Cloud SQL, choose one path:

1. Simple portfolio path:
   - Cloud SQL public IP.
   - Strong password.
   - Restricted access if possible.
   - Fastest to demo, but not the cleanest production setup.

2. Better GCP path:
   - Cloud SQL connection through Cloud Run connector.
   - Use `--add-cloudsql-instances`.
   - SQLAlchemy URL supports Cloud SQL Unix socket through `CLOUD_SQL_CONNECTION_NAME`.

Recommended final direction:

```env
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_DATABASE=demo_bot
CLOUD_SQL_CONNECTION_NAME=<project>:<region>:<instance>
```

### Worker

The current worker exists in `backend/docker-compose.yml` as:

```bash
celery -A tasks.celery_app worker --loglevel=info --concurrency=1
```

For Cloud Run worker:

- Build from the same backend image.
- Override command to run Celery.
- Connect to Memorystore Redis or another Redis/Valkey-compatible broker.
- Do not run worker in the core demo path unless document import is needed.

---

## 4. Environment Variables

### Backend - Core Demo

Store these in Cloud Run environment variables or Secret Manager:

```env
GEMINI_API_KEY=<secret>
GOOGLE_API_KEY=<secret>
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash

QDRANT_URL=<qdrant-cloud-url>
QDRANT_API_KEY=<secret>
QDRANT_COLLECTION=llm
QDRANT_VECTOR_SIZE=<embedding-dimension>
QDRANT_DISTANCE=COSINE

EMBEDDING_API_KEY=<secret>
EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
EMBEDDING_MODEL=gemini-embedding-001

MYSQL_USER=<cloud-sql-user>
MYSQL_PASSWORD=<secret>
MYSQL_DATABASE=demo_bot
MYSQL_HOST=<cloud-sql-host-or-socket-config>
MYSQL_PORT=3306

VIETNAMESE_LLM_API_URL=
CUSTOM_EMBEDDING_ENABLED=false
ENABLE_LLM_REPHRASE=false
ENABLE_LLM_ROUTER=false
ENABLE_LLM_QUERY_REWRITE=false
ENABLE_LLM_SUMMARY=false
TASK_TIMEOUT=420
```

### Frontend

```env
API_BASE_URL=<backend-cloud-run-url>
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_PORT=${PORT}
```

### Worker - Async Extension

```env
CELERY_BROKER_URL=redis://<memorystore-host>:6379/0
CELERY_RESULT_BACKEND=redis://<memorystore-host>:6379/1
GCS_BUCKET_NAME=<legal-documents-bucket>
QDRANT_URL=<qdrant-cloud-url>
QDRANT_API_KEY=<secret>
```

Do not commit real values to GitHub.

---

## 5. Deployment Phases

### Phase 0 - Clean Local Demo

Goal: make sure the local demo works before cloud deployment.

Checklist:

- Backend can answer via Gemini when `VIETNAMESE_LLM_API_URL=` is empty.
- Qdrant collection `llm` has indexed legal data.
- Frontend reads `API_BASE_URL` from environment variable.
- Chat response displays clean Markdown/HTML formatting.
- `.env` files are ignored by Git.

Local check:

```bash
curl http://localhost:8000/health
```

Open:

```text
http://localhost:8051
```

### Phase 1 - Prepare Cloud Resources

Use one GCP region, preferably:

```text
asia-southeast1
```

Resources:

- Artifact Registry repository.
- Cloud Run frontend service.
- Cloud Run backend service.
- Cloud SQL MySQL instance.
- Cloud Storage bucket for documents.
- Secret Manager secrets.
- Qdrant Cloud cluster.

Enable APIs:

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com
```

### Phase 2 - Build and Push Images

Pattern reused from previous Cloud Run projects:

```bash
PROJECT_ID=<your-gcp-project-id>
REGION=asia-southeast1
REPO_NAME=legal-rag-repo
ARTIFACT_HOST=$REGION-docker.pkg.dev
REPO_PATH=$ARTIFACT_HOST/$PROJECT_ID/$REPO_NAME
```

Create repository:

```bash
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker repository for Vietnamese Legal RAG Chatbot" \
  --project="$PROJECT_ID" \
  --quiet
```

Build backend:

```bash
gcloud builds submit backend \
  --tag "$REPO_PATH/legal-rag-backend:latest" \
  --project "$PROJECT_ID"
```

Build frontend:

```bash
gcloud builds submit frontend \
  --tag "$REPO_PATH/legal-rag-frontend:latest" \
  --project "$PROJECT_ID"
```

### Phase 3 - Deploy Backend Cloud Run

Deploy backend first because frontend needs its URL.

```bash
gcloud run deploy legal-rag-backend \
  --image "$REPO_PATH/legal-rag-backend:latest" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 10 \
  --timeout 420 \
  --min-instances 0 \
  --set-env-vars GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/,GEMINI_MODEL=gemini-2.5-flash,VIETNAMESE_LLM_API_URL=,CUSTOM_EMBEDDING_ENABLED=false \
  --project "$PROJECT_ID"
```

Secrets should be attached through Secret Manager, for example:

```bash
gcloud run services update legal-rag-backend \
  --region "$REGION" \
  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest,GOOGLE_API_KEY=GEMINI_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,MYSQL_PASSWORD=MYSQL_PASSWORD:latest \
  --project "$PROJECT_ID"
```

Get backend URL:

```bash
BACKEND_URL=$(gcloud run services describe legal-rag-backend \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')
```

Health check:

```bash
curl "$BACKEND_URL/health"
```

### Phase 4 - Deploy Frontend Cloud Run

```bash
gcloud run deploy legal-rag-frontend \
  --image "$REPO_PATH/legal-rag-frontend:latest" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 1Gi \
  --concurrency 20 \
  --timeout 300 \
  --min-instances 0 \
  --set-env-vars API_BASE_URL="$BACKEND_URL",STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  --project "$PROJECT_ID"
```

Get frontend URL:

```bash
FRONTEND_URL=$(gcloud run services describe legal-rag-frontend \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')
```

Open:

```bash
echo "$FRONTEND_URL"
```

### Phase 5 - Add Async Extension

Only add this after the core demo is stable.

Create Cloud Storage bucket:

```bash
gsutil mb -l "$REGION" "gs://<legal-documents-bucket>"
```

Deploy worker from backend image:

```bash
gcloud run deploy legal-rag-worker \
  --image "$REPO_PATH/legal-rag-backend:latest" \
  --platform managed \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 1 \
  --timeout 3600 \
  --command celery \
  --args "-A,tasks.celery_app,worker,--loglevel=info,--concurrency=1,--prefetch-multiplier=1" \
  --project "$PROJECT_ID"
```

Important note:

```text
Cloud Run services are request-driven. For a continuously running Celery worker, Cloud Run Jobs, GKE, Compute Engine, or a worker service with min instances may be more reliable.
```

For portfolio explanation, it is acceptable to describe this as an async extension plan unless the worker is fully deployed and tested.

---

## 6. Data and Indexing Plan

Do not store large legal documents inside the Cloud Run image.

Use:

```text
Cloud Storage
  -> raw documents
  -> processed JSONL/CSV
  -> import job
  -> Qdrant Cloud
```

Recommended flow:

```text
Upload documents to Cloud Storage
  -> backend creates indexing job
  -> worker downloads file
  -> clean/chunk text
  -> create embeddings
  -> upsert vectors to Qdrant Cloud
  -> save job status to Cloud SQL
```

For first demo:

- Pre-index a small verified legal dataset.
- Do not allow public users to upload arbitrary files yet.
- Show RAG retrieval quality using fixed legal questions.

---

## 7. Security and Cost Control

Security:

- Never commit `.env` files.
- Use Secret Manager for Gemini, Qdrant, Cloud SQL, and Redis credentials.
- Keep backend public only if needed for frontend demo.
- Keep worker private.
- Restrict Cloud SQL access.
- Do not expose Qdrant API key in frontend.

Cost:

- Keep Cloud Run `min-instances=0` for portfolio demo.
- Use Gemini API as default instead of self-hosted LLM.
- Use Qdrant Cloud free/small tier if available.
- Use small Cloud SQL instance.
- Run fine-tuned LLM only on demand through Vast.ai if needed.

---

## 8. Interview Story for FPT Telecom

Use this version:

```text
I designed the project as a deployable fullstack AI product. The core demo runs on Cloud Run with a Streamlit frontend, FastAPI backend, Qdrant Cloud for vector retrieval, Cloud SQL for app state, and Gemini API for stable LLM generation. For production-style document ingestion, I planned an async extension using Cloud Run worker, Celery, Redis-compatible broker, and Cloud Storage. The system also keeps optional support for my fine-tuned Vietnamese legal LLM through an OpenAI-compatible endpoint, but the default demo path remains lightweight and reliable.
```

Short CV bullet:

```text
Built a deployable Vietnamese legal RAG chatbot using Cloud Run, FastAPI, Streamlit, Qdrant Cloud, Cloud SQL, Docker, and Gemini API, with an async document-indexing extension using Celery, Redis-compatible broker, and Cloud Storage.
```

If asked why not deploy the fine-tuned LLM by default:

```text
For a public demo, Gemini API gives stable latency and lower cost. The fine-tuned LLM is kept as an optional provider through an OpenAI-compatible endpoint, mainly to demonstrate model customization and local/GPU serving experience.
```

---

## 9. Final Demo Scope

Must show:

- Public frontend URL.
- Backend health endpoint.
- At least 3 legal questions answered with retrieved context.
- Qdrant Cloud collection already indexed.
- Cloud SQL stores chat history or app records.
- README explains core demo and async extension clearly.

Do not show unless fully stable:

- Fine-tuned LLM running on GPU.
- Complex multi-tool agent flows.
- Monitoring stack.
- Public document upload.

Final positioning:

```text
Core demo: Cloud Run frontend + Cloud Run backend + Qdrant Cloud + Cloud SQL + Gemini API.
Async extension: Cloud Run worker + Celery + Memorystore Redis/Valkey-compatible broker + Cloud Storage.
Advanced optional: fine-tuned Vietnamese legal LLM via OpenAI-compatible endpoint.
```
