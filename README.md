# Mathematical Routing Agent with Agno

A comprehensive AI system for solving mathematical problems with step-by-step explanations, built with FastAPI, Agno, React, and Qdrant.

## Features

- **AI Gateway with Guardrails**: Input/output validation focused on mathematics-only queries
- **Vector Database Knowledge Base**: Using Qdrant with embeddings from sentence-transformers
- **Agno-based Math Agent**: Step-by-step mathematical problem solving and solution validation
- **Model Context Protocol (MCP) Server**: Advanced mathematical web search using Tavily
- **Human-in-the-Loop Feedback System**: Async review requests, feedback parsing, and continuous learning
- **JEE Benchmark Evaluation**: Assessing system accuracy on JEE Main 2025 dataset
- **React Frontend**: Interactive chat interface with inline feedback forms

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Chat Interface │  │ Feedback UI │  │ Admin Panel │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                          HTTP/WebSocket
                              │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ API Gateway │  │ Auth System │  │ Guardrails  │         │
│  │ Routing     │  │             │  │ Layer       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Math Agent  │  │ Knowledge   │  │ Web Search  │         │
│  │ (Agno)      │  │ Base RAG    │  │ MCP Server  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Human-Loop  │  │ Feedback    │  │ Evaluation  │         │
│  │ Manager     │  │ Processor   │  │ Engine      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Vector DB   │  │ PostgreSQL  │  │ Redis Cache │         │
│  │ (Qdrant)    │  │ (Metadata)  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Backend Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints/
│   │       └── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── guardrails.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── math_agent.py
│   ├── models/
│   │   └── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── feedback_service.py
│   │   ├── image_service.py
│   │   ├── mcp_service.py
│   │   ├── pdf_processing_service.py
│   │   └── vector_service.py
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_math_agent.py
│   └── utils/
│       ├── __init__.py
│       └── evaluation.py
├── knowledge_base/
│   ├── processed/
│   └── README.md
├── uploads/
├── Dockerfile
├── README.md
├── explore_agno.py
├── jee_benchmark_results.json
├── jee_benchmark_results_detailed.json
├── jee_benchmark_summary.json
├── run_benchmark.py
├── test_agno_basic.py
└── test_agno_integration.py
├── requirements.txt
└── docker-compose.yml
```

## Frontend Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx
│   │   ├── FeedbackForm.tsx
│   │   └── ProcessingLoader.tsx
│   ├── hooks/
│   │   └── useChat.ts
│   ├── services/
│   │   └── api.ts
│   ├── styles/
│   │   └── globals.css
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   └── main.tsx
├── Dockerfile
├── index.html
├── package-lock.json
├── package.json
└── vite.config.ts
```

## Setup and Installation

### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Frontend

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the Application

### Using Docker (Recommended)

```
docker-compose up --build
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup

1. Start the backend:
   ```
   cd backend
   uvicorn app.main:app --reload
   ```

2. Start the frontend:
   ```
   cd frontend
   npm run dev
   ```

## API Endpoints

- `POST /api/v1/solve` - Solve a mathematical problem
- `POST /api/v1/feedback` - Submit feedback on a solution
- `GET /api/v1/pending-reviews` - Get pending human reviews
- `POST /api/v1/submit-review` - Submit human review
- `GET /health` - Health check

## Development

### Backend Testing

Run the JEE benchmark evaluation:
```bash
cd backend
python run_benchmark.py
```

### Frontend Development

The React frontend uses Vite for fast development and hot reloading.


## Evaluation

`backend/evals/` has the evaluation tooling for the agent. It uses only the stdlib and pytest, and is covered by CI in `.github/workflows/evals.yml`. Full details are in [backend/evals/README.md](backend/evals/README.md).

- **pass@k.** The unbiased estimator from the Codex paper, `1 - C(n-c, k) / C(n, k)`, computed in its numerically stable form.
- **SWE-bench-style patch harness.** Tasks use SWE-bench's fields (`problem_statement`, gold `patch`, hidden `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`). A candidate patch is applied in a temp copy of the repo and the tests are run. The task counts as **resolved** only if all FAIL_TO_PASS and PASS_TO_PASS tests pass. There are three local tasks, adapted from real bugs in this backend, and the gold patches double as a self-check of the harness.
- **Model runner.** Samples n patches per task from a model (a deterministic mock for CI, or Claude via `ANTHROPIC_API_KEY`) and writes pass@1 and pass@k reports as JSON and Markdown.
- **SWE-bench Lite / Verified adapter.** Loads real instances from Hugging Face for a dry run and writes predictions in the official harness format. Scoring them requires the official Docker-based `swebench` harness, which is not run here.
- **Rubric review.** Scores free-text answers 1-5 on correctness, instruction following, grounding (hallucination), formatting and safety, each with a rationale. Supports an LLM-as-judge mode, human review through CSV, and judge-vs-human agreement via Cohen's kappa (plain and quadratic-weighted).

```bash
cd backend
pip install pytest
python -m pytest evals -q
python -m evals validate
python -m evals run --model mock -n 10 -k 1 5   # mock run: tests the pipeline, not a model score
```
