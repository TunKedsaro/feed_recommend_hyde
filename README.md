# 📄 HyDE Generator API
A production-ready HyDE (Hypothetical Document Embedding) generation service built with FastAPI, LLM (Google Gemini), and vector embedding pipelines.

This service generates semantic search queries (HyDE queries) for each user (student), converts them into embeddings, and stores results in Google Cloud Storage (GCS) for downstream retrieval systems.

---
# 🚀 Key Capabilities
- Generate personalized HyDE queries from user data
- LLM-based semantic query generation (Gemini)
- Multi-query generation per user (e.g. 5 queries/student)
- Embedding generation for each query
- GCS-based storage (embedding + metadata)
- Config-driven pipeline (YAML)
- Supports Sequential and single-user processing (FIG)
- Cloud-native (Cloud Run ready)
- Designed for offline pipeline (no LLM in serving path)

--- 
# 🧠 Architecture Overview
Overview 
BigQuery -> LLM -> GCS

High-level pipeline
BigQuery (students, interactions, feeds)
↓
Context Builder
↓
Prompt Builder (YAML-driven)
↓
LLM (HyDE Query Generation)
↓
Embedding Model
↓
GCS Storage (5embedding + 1hyde_json + 1metadata)

Flow per student:
student_id
↓
Build context (profile + history)
↓
Generate HyDE queries (LLM)
↓
Convert queries -> embeddings
↓
Upload to GCS

---
# 📦 Output Artifact (Core Concept)
Each student produces a single deterministic bundle
gs://hyde-datalake-test/{student_id}/hyde_bundle.json

Structure
{
  "student_id": "string",
  "metadata": {...},
  "hyde_queries": [...],
  "embeddings": {...}
}

Design Decisions
- ✅ 1 bundle per student
- ✅ 1 GCS upload per student
- ❌ No fragmented uploads (performance optimized)
- ✅ Embeddings precomputed for fast retrieval

---
# 📁 Project Structure
```
/code
├── Dockerfile.dev
├── Dockerfile.prod
├── cloudbuild.yaml
├── requirements.txt
├── README.md
├── src
│   ├── main.py                # FastAPI entrypoint
│   ├── functions
│   │   ├── core               # Business logic (HyDE pipeline)
│   │   │   ├── hydegenerator.py
│   │   │   ├── context_builder.py
│   │   │   └── history.py
│   │   └── utils              # External integrations
│   │       ├── llm_client.py
│   │       ├── text_embeddings.py
│   │       ├── shin_embedder.py
│   │       ├── bigquery.py
│   │       └── cloudstorage.py
│   └── parameters             # Config (YAML-driven)
│       ├── parameters.yaml
│       ├── prompts.yaml
│       └── retrieval_score_weights.yaml
├── tests                      # Unit tests
│   └── test_gcs_bundle.py
├── docs                       # Design / API docs
├── secrets                    # Local credentials (DO NOT COMMIT)
└── sandbox                    # Experiment (not production)
```

---
# ⚙️ Configuration System (YAML-Driven)
parameters.yaml
- LLM configuration (model, temperature, tokens)
- HyDE generation rules (history usage, retry, truncation)
prompts.yaml
- Prompt templates (A / B / C strategies)
- BigQuery + GCS config
- Query structure enforcement
config.yaml
- External system configuration
- Dataset & bucket mapping
---

# 🧠 HyDE Query Design
Each student generates exactly 5 queries
Query Intent
Q1 role_or_skill
Q2 history_aligned
Q3 practical
Q4 exploratory
Q5 exploratory
Rules:
- Deterministic slot assignment
- JSON-only output
- Language-aware generation
- Search-style queries (not sentences)
---

# 📡 API Endpoints
Base path:
/hyde

Health Check
GET /health
---

### Generate HyDE (Single Student)
```
POST /hyde/students/{student_id}
Response:
{
  "student_id": "stu_p001",
  "status": "complete",
  "total_time_sec": 12.4
}
```

### Generate HyDE (Sequential)
```
GET /hyde/students/sequential
```
---

# 🐳 Running Locally
Install dependencies
```
pip install -r requirements.txt
```

Run API
```
uvicorn src.main:app --host 0.0.0.0 --port 4000 --reload
```

Health check
```
curl http://localhost:4000/health
```

Swagger UI
```
http://localhost:4000/docs
```
---

# ☁️ Deployment (GCP)
Stack
- Cloud Run (compute)
- Cloud Build (CI/CD)
- Artifact Registry (images)
- BigQuery (input)
- GCS (artifact storage)

```
### Build
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=poc-piloturl-nonprod

### Deploy
gcloud run deploy hyderecomment-service \
  --image="asia-southeast1-docker.pkg.dev/poc-piloturl-nonprod/hyderecomment/hyderecomment:latest" \
  --region="asia-southeast1" \
  --port=4000 \
  --memory=8Gi \
  --cpu=4 \
  --max-instances=10 \
  --min-instances=1 \
  --service-account=hyde-bq-sa@poc-piloturl-nonprod.iam.gserviceaccount.com \
  --set-env-vars="APP_ENV=dev,GOOGLE_API_KEY=<API_KEY>" \
  --allow-unauthenticated
```
---

### Environment Variables
Variable Description
APP_ENV dev / prod
GOOGLE_API_KEY Gemini + Embedding API

---
# 📊 Observability
Per-student metrics:
- download_data_ms
- build_context_ms
- build_prompt_ms
- llm_call_ms
- embedding_ms
- upload_gcs_ms
- total_ms

---
# ⚡ Performance Summary
Optimized Pipeline
Stage Time
- LLM ~3.1 sec
- Context build ~2.4 sec
- GCS upload ~0.9 sec
- Total: ~6–7 sec / student

Scaling Estimate
- Students Runtime
- 100 ~10–12 min
- 1,000 ~100–120 min
---

# 🧪 Key Optimizations
✅ BigQuery
- Download once per batch
- Avoid per-student queries
✅ GCS Upload
- 1 bundle upload (not 7 uploads)
- Reduced latency from ~4.6s → ~0.9s
---

# ⚠️ Known Limitations
- LLM latency dominates pipeline
- Sequential execution (no parallel yet)
- Embedding API concurrency limits
---

# 🔧 Future Improvements
- Parallel student processing
- Async LLM + embedding calls
- Distributed batch workers
- Pre-cached embeddings
--- 

# 🧠 Design Principles
- Deterministic outputs (CI-friendly)
- Offline-first architecture
- No LLM dependency in serving layer
- Reproducible pipelines
- Modular + YAML-driven system
--- 

# 🤝 Integration with Retrieval System
HyDE Generator produces:
hyde_bundle.json
Consumed by:
👉 Retrieval system (Shin)

```
Flow:
HyDE (GCS)
   ↓
Vector Index (Vertex AI shin)
   ↓
Subscore Calculation
   ↓
Top-K Feed Recommendation
```

# 📌 Summary
HyDE Generator is a core upstream component of the recommendation system:
- Converts user behavior → semantic search intent
- Enables fast, scalable vector retrieval
- Decouples expensive LLM calls from real-time serving
---

prod : https://hyderecomment-service-du7yhkyaqq-as.a.run.app/docs#/

dev  : https://hyderecomment-service-dev-du7yhkyaqq-as.a.run.app/docs#/Fetch%20results/get_student_feed_hyde_students__student_id__feed2_post