# CV Core Profile - Nguyen Khac Bao

This file is the source-of-truth for creating role-specific CVs. Keep claims
defensible, concise, and aligned with the target role. Do not paste API keys,
tokens, secrets, private URLs, or raw datasets into any CV.

## Fixed Profile

- Name: Nguyen Khac Bao
- Location: Ho Chi Minh City, Vietnam
- Email: nguyenkhacbaowork564@gmail.com
- Phone: (+84) 039 586 4429
- GitHub: https://github.com/NguyenKhacBao564
- Education: Posts and Telecommunications Institute of Technology (PTIT), Bachelor of Information Technology, 2022-2027
- Activities: AI Research Group - University Club, Member, 2025-Present
- Language: Vietnamese native; English technical reading proficiency for documentation, model cards, API references, papers, and engineering articles.

## Core Positioning

Primary positioning:

> AI integration intern who can turn AI models into usable demos and services
> with backend APIs, RAG, vector databases, Docker, and cloud deployment.

Avoid positioning as only a Computer Vision Engineer unless the job is clearly
CV-focused. The strongest current story is "deployable AI product + backend/API
+ cloud demo ability", with CV projects used as supporting evidence.

## Main Target Roles

### 1. Fullstack AI Integration Intern

Best project order:

1. Vietnamese Legal RAG Chatbot
2. Real-Time Face Mask Compliance Detection
3. Traffic Violation Detection & License Plate Recognition
4. Autofall Labeler Tool only if space remains

Highlight:

- FastAPI backend and Streamlit/browser frontend
- REST/WebSocket API integration
- Gemini API and OpenAI-compatible endpoints
- Qdrant Cloud, Cloud SQL, environment-based config, Secret Manager
- Docker, Cloud Run, Cloud Build, Artifact Registry
- Deployable demo, not "production-ready platform"

Avoid:

- Too much YOLO/CV detail in the top half
- Overclaiming legal correctness or production readiness
- Deep fine-tuning details unless asked

### 2. AI Engineer Intern - RAG / GenAI

Best project order:

1. Vietnamese Legal RAG Chatbot
2. Autofall Labeler Tool or Face Mask depending on JD
3. Traffic Violation Detection

Highlight:

- RAG pipeline, retrieval, embeddings, vector DB
- Gemini as default LLM provider
- Optional fine-tuned Vietnamese legal LLM via OpenAI-compatible endpoint
- Data import/indexing pipeline
- Backend service design and cloud-ready configuration

Use cautious wording:

- "RAG product demo"
- "cloud-ready"
- "optional extension"
- "deployed demo"
- "grounded answers using retrieved context"

Avoid:

- "Production legal AI"
- "Guaranteed legal advice"
- "Fully autonomous agent"

### 3. AI/ML/Data Engineer Intern

Best project order:

1. Vietnamese Legal RAG Chatbot
2. Traffic Violation Detection
3. Real-Time Face Mask Compliance Detection
4. Autofall Labeler Tool if the JD mentions annotation/data quality

Highlight:

- Data ingestion, JSONL/CSV processing, indexing
- Qdrant collection indexing and metadata payloads
- Cloud SQL chat history
- Dataset QA, manifests, train/val/test split
- SQL basics, Python, Pandas, Docker, Cloud Run

Avoid:

- Listing too many unpracticed data-stack tools
- Claiming BigQuery/Looker/PowerBI experience unless actually used

### 4. Computer Vision / AI Inference Intern

Best project order:

1. Real-Time Face Mask Compliance Detection
2. Traffic Violation Detection & License Plate Recognition
3. Autofall Labeler Tool
4. Vietnamese Legal RAG Chatbot as additional fullstack AI project

Highlight:

- FastAPI REST + WebSocket inference
- Browser webcam frontend
- Docker + Cloud Run HTTPS/WSS deployment
- Latency/FPS benchmark
- YOLOv8n, OpenCV, dataset QA, OCR, tracking

Avoid:

- Making the CV only about model metrics
- Claiming real-world robustness beyond tested data

## Project Bank

### Vietnamese Legal RAG Chatbot

Recommended title:

Vietnamese Legal RAG Chatbot

Recommended tech line:

Python, FastAPI, Streamlit, Gemini API, Qdrant Cloud, Cloud SQL, Docker, Google Cloud Run

Defensible bullets:

- Built a cloud-ready fullstack RAG product demo with a Streamlit frontend and FastAPI backend deployed as separate Google Cloud Run services.
- Integrated Gemini as the default LLM and embedding provider; indexed the full 32,980-record Vietnamese legal QA dataset into Qdrant Cloud for retrieval-based answers.
- Used Cloud SQL/MySQL for chat history and environment-based configuration with Secret Manager for API keys and database credentials.
- Designed optional extensions for async document indexing with Celery/Redis-compatible broker and fine-tuned Vietnamese legal LLM serving through an OpenAI-compatible endpoint.

Interview notes:

- Current demo flow: Streamlit -> FastAPI -> Cloud SQL history -> Gemini embedding -> Qdrant retrieval -> Gemini answer -> Cloud SQL save -> Streamlit.
- Fine-tuned LLM is optional and should not be the main claim for deployment roles.
- Celery/Redis is an extension path, not the current Cloud Run core demo path.

### Real-Time Face Mask Compliance Detection System

Recommended title:

Real-Time Face Mask Compliance Detection System

Recommended tech line:

Python, FastAPI, WebSocket, YOLOv8n, OpenCV, Docker, Google Cloud Run

Defensible bullets:

- Built and deployed an AI inference service with FastAPI REST + WebSocket endpoints and a browser webcam frontend running over Cloud Run HTTPS/WSS.
- Containerized the service for Cloud Run and logged warm endpoint latency around 151 ms for REST and 127 ms for WSS in the demo environment.
- Trained a YOLOv8n 3-class mask model on PWMFD converted from VOC to YOLO format: 9,205 images, 18,528 labeled instances, 0.970 mAP@50 validation.
- Added dataset audit scripts for split statistics, class imbalance, malformed labels, and reproducible YOLO training configuration.

Interview notes:

- Present this as an AI inference service, not just YOLO training.
- Mention that metrics are validation/demo-environment numbers, not a real-world SLA.

### Traffic Violation Detection & License Plate Recognition

Recommended title:

Traffic Violation Detection & License Plate Recognition

Recommended tech line:

Python, OpenCV, Ultralytics YOLO, ByteTrack, HyperLPR3, JSON

Defensible bullets:

- Built an AI video analytics pipeline for vehicle detection/tracking, traffic-light state estimation, stop-line event logic, plate OCR, and JSON evidence bundles.
- Prepared a compact CCPD2019 plate dataset with annotation parsing, YOLO bbox conversion, train/val/test split, negative samples, OCR crops, and QA reports.
- Fine-tuned a YOLO plate detector on 8,598 images, achieving 0.989 precision, 0.998 recall, and 0.994 mAP50 on validation.

Interview notes:

- Keep this shorter for Fullstack AI Integration roles.
- Be careful: current event precision is under configured stop-line rules, not a legal enforcement metric.

### Autofall Labeler Tool

Recommended title:

Autofall Labeler Tool

Recommended tech line:

Python, Streamlit, OpenCV, YOLO-Pose, Pandas

Defensible bullet:

- Built a human-in-the-loop dataset assistant with Streamlit review UI, metadata/annotation management, YOLO-Pose keypoints, QA reports, and reviewed-only export.
- Designed CSV-based source-of-truth records for annotations, metadata, keypoints, and review events to handle noisy CCTV data and reproducible dataset versioning.
- Achieved 90.2% coarse fall/non-fall agreement on a reviewed benchmark while identifying exact action-label accuracy as a limitation of weak temporal modeling.

When to include:

- Include for data labeling, dataset QA, annotation tooling, or Computer Vision roles.
- Remove for one-page Fullstack AI Integration CV if space is tight.

## CV Writing Rules

- One page unless specifically asked otherwise.
- Put the most relevant project first.
- Keep skills compact and only list tools that can be explained in an interview.
- Use "basic" when experience is basic.
- Use measured claims only when there is a report, deployed URL, or code evidence.
- Prefer "deployed demo", "cloud-ready", "optional extension", "validation metric".
- Avoid "production-ready", "enterprise-scale", "fully autonomous", and legal-advice overclaims.

## Current Latest CV

Keep only these as the current generated CV files:

- CV_Nguyen_Khac_Bao_FPT_Fullstack_AI_Integration_Intern.tex
- CV_Nguyen_Khac_Bao_FPT_Fullstack_AI_Integration_Intern.pdf
