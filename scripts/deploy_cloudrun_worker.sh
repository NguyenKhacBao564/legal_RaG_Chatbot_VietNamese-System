#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-asia-southeast1}"
REPO_NAME="${REPO_NAME:-legal-rag-repo}"
BACKEND_SERVICE="${BACKEND_SERVICE:-legal-rag-backend}"
WORKER_SERVICE="${WORKER_SERVICE:-legal-rag-worker}"
TAG="${TAG:-latest}"

CPU="${CPU:-2}"
MEMORY="${MEMORY:-2Gi}"
TIMEOUT="${TIMEOUT:-3600}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required. Run: export PROJECT_ID=<your-gcp-project-id>" >&2
  exit 1
fi

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$BACKEND_SERVICE:$TAG"

echo "Deploying async worker from image: $IMAGE"
echo "This assumes the backend image already exists and Celery broker env/secrets are configured."

gcloud run deploy "$WORKER_SERVICE" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --concurrency 1 \
  --timeout "$TIMEOUT" \
  --min-instances "$MIN_INSTANCES" \
  --command celery \
  --args "-A,tasks.celery_app,worker,--loglevel=info,--concurrency=1,--prefetch-multiplier=1" \
  --project "$PROJECT_ID"

echo "Worker deployed. For a continuously running worker, consider Cloud Run Jobs, GKE, or Compute Engine if Cloud Run scale-to-zero interrupts processing."
