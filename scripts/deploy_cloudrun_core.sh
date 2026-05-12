#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-asia-southeast1}"
REPO_NAME="${REPO_NAME:-legal-rag-repo}"
BACKEND_SERVICE="${BACKEND_SERVICE:-legal-rag-backend}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-legal-rag-frontend}"
TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"

BACKEND_CPU="${BACKEND_CPU:-2}"
BACKEND_MEMORY="${BACKEND_MEMORY:-2Gi}"
BACKEND_CONCURRENCY="${BACKEND_CONCURRENCY:-10}"
BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-420}"
BACKEND_MIN_INSTANCES="${BACKEND_MIN_INSTANCES:-0}"

FRONTEND_CPU="${FRONTEND_CPU:-1}"
FRONTEND_MEMORY="${FRONTEND_MEMORY:-1Gi}"
FRONTEND_CONCURRENCY="${FRONTEND_CONCURRENCY:-20}"
FRONTEND_TIMEOUT="${FRONTEND_TIMEOUT:-300}"
FRONTEND_MIN_INSTANCES="${FRONTEND_MIN_INSTANCES:-0}"

QDRANT_COLLECTION="${QDRANT_COLLECTION:-llm}"
QDRANT_VECTOR_SIZE="${QDRANT_VECTOR_SIZE:-3072}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-gemini-embedding-001}"
BACKEND_ENV_VARS="GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/,GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash},VIETNAMESE_LLM_API_URL=,CUSTOM_EMBEDDING_ENABLED=false,FORCE_SYNC_CHAT=true,QDRANT_COLLECTION=$QDRANT_COLLECTION,QDRANT_VECTOR_SIZE=$QDRANT_VECTOR_SIZE,EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/,EMBEDDING_MODEL=$EMBEDDING_MODEL"
CLOUD_SQL_ARGS=()

if [[ -n "${QDRANT_URL:-}" ]]; then
  BACKEND_ENV_VARS="$BACKEND_ENV_VARS,QDRANT_URL=$QDRANT_URL"
fi

if [[ -n "${MYSQL_USER:-}" ]]; then
  BACKEND_ENV_VARS="$BACKEND_ENV_VARS,MYSQL_USER=$MYSQL_USER"
fi

if [[ -n "${MYSQL_DATABASE:-}" ]]; then
  BACKEND_ENV_VARS="$BACKEND_ENV_VARS,MYSQL_DATABASE=$MYSQL_DATABASE"
fi

if [[ -n "${MYSQL_HOST:-}" ]]; then
  BACKEND_ENV_VARS="$BACKEND_ENV_VARS,MYSQL_HOST=$MYSQL_HOST"
fi

if [[ -n "${MYSQL_PORT:-}" ]]; then
  BACKEND_ENV_VARS="$BACKEND_ENV_VARS,MYSQL_PORT=$MYSQL_PORT"
fi

if [[ -n "${CLOUD_SQL_CONNECTION_NAME:-}" ]]; then
  BACKEND_ENV_VARS="$BACKEND_ENV_VARS,CLOUD_SQL_CONNECTION_NAME=$CLOUD_SQL_CONNECTION_NAME"
  CLOUD_SQL_ARGS+=(--add-cloudsql-instances "$CLOUD_SQL_CONNECTION_NAME")
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required. Run: export PROJECT_ID=<your-gcp-project-id>" >&2
  exit 1
fi

ARTIFACT_HOST="$REGION-docker.pkg.dev"
REPO_PATH="$ARTIFACT_HOST/$PROJECT_ID/$REPO_NAME"
BACKEND_IMAGE="$REPO_PATH/$BACKEND_SERVICE:$TAG"
FRONTEND_IMAGE="$REPO_PATH/$FRONTEND_SERVICE:$TAG"

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Repository: $REPO_NAME"
echo "Backend image: $BACKEND_IMAGE"
echo "Frontend image: $FRONTEND_IMAGE"

echo "Enabling required Google Cloud APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  --project "$PROJECT_ID"

echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker repository for Vietnamese Legal RAG Chatbot" \
  --project="$PROJECT_ID" \
  --quiet || true

echo "Building backend image..."
gcloud builds submit backend \
  --tag "$BACKEND_IMAGE" \
  --project "$PROJECT_ID"

echo "Building frontend image..."
gcloud builds submit frontend \
  --tag "$FRONTEND_IMAGE" \
  --project "$PROJECT_ID"

echo "Deploying backend..."
gcloud run deploy "$BACKEND_SERVICE" \
  --image "$BACKEND_IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu "$BACKEND_CPU" \
  --memory "$BACKEND_MEMORY" \
  --concurrency "$BACKEND_CONCURRENCY" \
  --timeout "$BACKEND_TIMEOUT" \
  --min-instances "$BACKEND_MIN_INSTANCES" \
  "${CLOUD_SQL_ARGS[@]}" \
  --set-env-vars "$BACKEND_ENV_VARS" \
  --project "$PROJECT_ID"

echo "Attach secrets manually if they are not already configured:"
echo "  gcloud run services update $BACKEND_SERVICE --region $REGION --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest,GOOGLE_API_KEY=GEMINI_API_KEY:latest,EMBEDDING_API_KEY=GEMINI_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,MYSQL_PASSWORD=MYSQL_PASSWORD:latest --project $PROJECT_ID"

BACKEND_URL="$(gcloud run services describe "$BACKEND_SERVICE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')"

echo "Backend URL: $BACKEND_URL"
echo "Backend health check:"
curl -fsS "$BACKEND_URL/health"
echo

echo "Deploying frontend..."
gcloud run deploy "$FRONTEND_SERVICE" \
  --image "$FRONTEND_IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu "$FRONTEND_CPU" \
  --memory "$FRONTEND_MEMORY" \
  --concurrency "$FRONTEND_CONCURRENCY" \
  --timeout "$FRONTEND_TIMEOUT" \
  --min-instances "$FRONTEND_MIN_INSTANCES" \
  --set-env-vars API_BASE_URL="$BACKEND_URL",CHAT_SYNC_REQUEST=true,STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  --project "$PROJECT_ID"

FRONTEND_URL="$(gcloud run services describe "$FRONTEND_SERVICE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')"

echo "Frontend URL: $FRONTEND_URL"
