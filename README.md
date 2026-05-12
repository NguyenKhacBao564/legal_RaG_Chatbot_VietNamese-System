# Vietnamese Legal Chatbot RAG System

He thong chatbot tu van phap luat Viet Nam su dung FastAPI, Streamlit, Celery,
MariaDB, Valkey, Qdrant va LLM OpenAI-compatible. Du an co the chay voi Gemini
API hoac model local fine-tuned cua ban qua endpoint `/v1/chat/completions`.

## Tinh nang chinh

- Giao dien chat bang Streamlit.
- Backend FastAPI cho chat, health check, import tai lieu va tao collection.
- Celery worker xu ly tac vu bat dong bo.
- Luu lich su hoi thoai bang MariaDB.
- Vector database Qdrant cho RAG.
- Ho tro endpoint LLM OpenAI-compatible, mac dinh dung Gemini.
- Ho tro embedding service rieng, phu hop chay tren Vast.ai GPU.
- Agent tools cho mot so tac vu phap ly nhu tinh phat hop dong, chia thua ke,
  kiem tra tuoi phap ly va tim kiem web.

## Cau truc thu muc

```text
.
├── backend/                  # FastAPI API, Celery tasks, RAG pipeline
├── frontend/                 # Streamlit UI
├── database/                 # MariaDB docker compose va init script
├── embed_serving/            # Embedding service deployment
├── llm_finetuning_serving/   # Fine-tuning va serving LLM rieng
├── data_pipeline/            # Xu ly du lieu training/RAG
├── docs/                     # Tai lieu ky thuat
├── tests/                    # Test suite
└── asset/                    # Hinh anh/demo local, video lon khong commit
```

## Yeu cau

- Docker va Docker Compose v2.
- Gemini API key hoac mot endpoint LLM OpenAI-compatible.
- Tuy chon: Vast.ai GPU de fine-tune/chay model nang.
- Tuy chon tren Mac Apple Silicon: `llama-cpp-python` de chay model local GGUF.

## Cau hinh moi truong

Tao Docker network dung chung:

```bash
docker network create internal-network
```

Tao file `database/.env`:

```env
MYSQL_ROOT_PASSWORD=change_me
```

Tao file `backend/.env` tu `backend/.env.template`, toi thieu can cac bien sau:

```env
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_ROOT_PASSWORD=change_me
MYSQL_DATABASE=demo_bot
MYSQL_HOST=mariadb-tiny
MYSQL_PORT=3306
CLOUD_SQL_CONNECTION_NAME=

CELERY_BROKER_URL=redis://valkey-db:6379
CELERY_RESULT_BACKEND=redis://valkey-db:6379

GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=your_gemini_api_key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-2.5-flash

QDRANT_URL=http://qdrant-db:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=llm
QDRANT_VECTOR_SIZE=1024

EMBEDDING_API_KEY=your_gemini_api_key
EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
EMBEDDING_MODEL=gemini-embedding-001

CUSTOM_EMBEDDING_API_URL=http://your-embedding-host:5001
CUSTOM_EMBEDDING_ENABLED=true

# De trong neu muon dung Gemini truc tiep.
# Dat endpoint nay neu chay fine-tuned LLM local/Vast.ai.
VIETNAMESE_LLM_API_URL=
LOCAL_LLM_TIMEOUT=420
LOCAL_LLM_MAX_TOKENS=384
ENABLE_REMOTE_FALLBACK=false
ENABLE_LLM_REPHRASE=false
ENABLE_LLM_ROUTER=false
ENABLE_LLM_QUERY_REWRITE=false
ENABLE_LLM_SUMMARY=false
FORCE_SYNC_CHAT=false
TASK_TIMEOUT=420
```

Khong commit cac file `.env`. Chung da duoc ignore.

## Cloud Run core demo

Core deploy path cho portfolio/interview:

```text
Cloud Run frontend + Cloud Run backend + Qdrant Cloud + Cloud SQL + Gemini API
```

Repo da co cac thay doi can thiet cho Cloud Run:

- Backend doc bien `PORT` cua Cloud Run.
- Frontend doc `API_BASE_URL` thay vi hard-code Docker hostname.
- Qdrant doc `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.
- Query-time embedding co the dung custom embedding service hoac Gemini/OpenAI-compatible embedding API.
- Backend co `FORCE_SYNC_CHAT=true` de demo core khong bat buoc chay Celery worker.
- Cloud SQL co the cau hinh qua `MYSQL_*` hoac `CLOUD_SQL_CONNECTION_NAME`.

Luu y: embedding model dung luc import/index va luc query phai giong nhau. Neu doi tu custom embedding sang Gemini embedding, can tao lai collection Qdrant voi `QDRANT_VECTOR_SIZE` phu hop va import lai vector.

Script build/deploy mau:

```bash
export PROJECT_ID=<your-gcp-project-id>
export REGION=asia-southeast1
export QDRANT_URL=<your-qdrant-cloud-url>
export QDRANT_VECTOR_SIZE=3072
export MYSQL_USER=<cloud-sql-user>
export MYSQL_DATABASE=demo_bot
export CLOUD_SQL_CONNECTION_NAME=<project>:<region>:<instance>

./scripts/deploy_cloudrun_core.sh
```

Secrets nen dat bang Secret Manager, khong ghi vao Git:

```bash
gcloud run services update legal-rag-backend \
  --region "$REGION" \
  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest,GOOGLE_API_KEY=GEMINI_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,MYSQL_PASSWORD=MYSQL_PASSWORD:latest \
  --project "$PROJECT_ID"
```

Async extension sau khi core demo on dinh:

```text
Cloud Run worker + Celery + Memorystore Redis/Valkey-compatible broker + Cloud Storage
```

Script mau:

```bash
./scripts/deploy_cloudrun_worker.sh
```

## Chay model local tren Mac

May local 16GB RAM khong nen chay Llama 8B bang Transformers FP16 vi rat cham
va de bi swap. Duong chay local nen dung GGUF Q4 voi `llama.cpp`.

File model lon khong nam trong Git. Tren may nay da chuan bi:

```text
llm_finetuning_serving/models/gguf/Llama-3.1-8B-Instruct-Q4_K_M.gguf
llm_finetuning_serving/models/gguf/vietnamese-legal-lora.gguf
```

Neu can tao lai, tai base GGUF `unsloth/Llama-3.1-8B-Instruct-GGUF` ban
`Q4_K_M`, sau do convert adapter PEFT `NguyenBao564/vietnamese-legal-llama-3.1-8b`
sang GGUF bang `convert_lora_to_gguf.py` cua llama.cpp. Khong commit cac file
`.gguf`.

Chay server model local:

```bash
cd llm_finetuning_serving/serving
python serve_model_gguf.py
```

Kiem tra:

```bash
curl http://localhost:6000/health
```

Trong `backend/.env`, tro backend Docker ve model local:

```env
VIETNAMESE_LLM_API_URL=http://host.docker.internal:6000/v1/chat/completions
ENABLE_REMOTE_FALLBACK=false
```

## Chay bang Docker

Chay database:

```bash
cd database
docker compose up -d
```

Chay backend:

```bash
cd backend
docker compose up -d --build
```

Chay frontend:

```bash
cd frontend
docker compose up -d --build
```

URL mac dinh:

- Frontend: <http://localhost:8051>
- Backend health: <http://localhost:8000/health>
- Qdrant: <http://localhost:6333>
- MariaDB local port: `3308`

Kiem tra nhanh backend:

```bash
curl http://localhost:8000/health
```

Gui thu chat dong bo:

```bash
curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "bot_id": "botLawyer",
    "user_message": "Xin chao, ban co the giup gi?",
    "sync_request": true
  }'
```

## Du lieu RAG

Repo khong commit dataset lon. Neu muon dung RAG that su, dat file JSONL tai:

```text
backend/data/train.jsonl
```

Moi dong co dang:

```json
{"question":"...","context":"..."}
```

Import vao Qdrant:

```bash
docker exec -it chatbot-api python src/import_data.py --data-file /usr/src/app/data/train.jsonl
```

Neu chua import dataset hoac embedding service chua san sang, backend van co the
tra loi bang Gemini fallback, nhung cau tra loi khong dua tren kho tai lieu cuc
bo.

## Chay voi Vast.ai

Vast.ai nen dung cho hai muc dich:

1. Chay embedding service va tro `CUSTOM_EMBEDDING_API_URL` ve public URL cua
   instance.
2. Chay fine-tuned LLM OpenAI-compatible va dat `VIETNAMESE_LLM_API_URL`.

Khong bat buoc fine-tune de chay du an. Fine-tune chi can khi muon thay Gemini
bang model rieng.

## Lenh quan tri huu ich

```bash
docker ps
docker logs -f chatbot-api
docker logs -f chatbot-worker
docker logs -f chatbot-ui
docker compose down
```

## Bao mat va file khong commit

`.gitignore` da chan:

- `.env`, token, certificate, private key.
- Dataset, file JSONL/parquet/pickle.
- Model checkpoint va artifact lon.
- Docker volume, database local, cache.
- Video/gif demo trong `asset/`.

Truoc khi push, nen kiem tra:

```bash
git status --short
git ls-files | grep -E '(.env|.mkv|.mp4|.gif|.safetensors|.gguf|.pt|.pth)$'
```

## License

Them license phu hop voi muc dich su dung cua ban truoc khi phat hanh public.
