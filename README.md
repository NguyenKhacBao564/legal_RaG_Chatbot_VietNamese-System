# Vietnamese Legal Chatbot RAG System

He thong chatbot tu van phap luat Viet Nam su dung FastAPI, Streamlit, Celery,
MariaDB, Valkey, Qdrant va LLM OpenAI-compatible. Ban hien tai duoc cau hinh de
chay voi Gemini API, co fallback tra loi truc tiep bang Gemini khi chua import
du lieu RAG.

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
- Tuy chon: Vast.ai GPU de chay embedding service hoac fine-tuned LLM.

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
MYSQL_ROOT_PASSWORD=change_me
MYSQL_HOST=mariadb-tiny
MYSQL_PORT=3306

CELERY_BROKER_URL=redis://valkey-db:6379
CELERY_RESULT_BACKEND=redis://valkey-db:6379

GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=your_gemini_api_key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-2.5-flash

CUSTOM_EMBEDDING_API_URL=http://your-embedding-host:5001
CUSTOM_EMBEDDING_ENABLED=true

# De trong neu muon dung Gemini truc tiep.
# Dat endpoint nay neu da deploy fine-tuned LLM tren Vast.ai.
VIETNAMESE_LLM_API_URL=
```

Khong commit cac file `.env`. Chung da duoc ignore.

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
