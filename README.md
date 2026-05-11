# Business Research Assistant

A multi-agent business research assistant built with **LangGraph**, **Groq** (llama-3.3-70b-versatile), and **Tavily**. Type any business question; the system clarifies it if vague, runs targeted web searches, validates the findings, then writes you a structured report. Comes with both an interactive CLI and a Next.js web UI.

## Pipeline

```
START
  |
  v
clarity_agent  ── needs_clarification ──>  human_feedback  (pauses for input)
   ^                                              |
   └──────────────── loops back ─────────────────┘
   |
 clear
   |
   v
research_agent ──> validator_agent
                       |
                       ├── insufficient + attempts < 3 ──> research_agent (retry)
                       |
                       └── sufficient OR max attempts ──> synthesis_agent ──> END
```

| Agent | Responsibility |
| --- | --- |
| `clarity_agent` | Decides if the query has a named entity and a focused intent. |
| `human_feedback` | Pauses the graph via `interrupt()` to ask the user for clarification. |
| `research_agent` | Runs 3 targeted Tavily searches; synthesises raw results + a 0–10 confidence score. |
| `validator_agent` | Gate: are findings relevant, complete, and substantial? Returns `sufficient` / `insufficient`. |
| `synthesis_agent` | Writes the final markdown report (Overview / Recent / Financial / Takeaways). Adds a data-quality disclaimer when confidence is low. |

## Project layout

Repository root (`synapse-ai/`):

```
.
├── .env                 # your real API keys (gitignored)
├── .env.example         # template
├── README.md
├── requirements.txt
├── backend/
│   ├── main.py          # interactive CLI runner
│   ├── api.py           # FastAPI + SSE for the web UI
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── graph_builder.py
│   │   └── nodes/
│   │       ├── clarity_agent.py
│   │       ├── research_agent.py
│   │       ├── validator_agent.py
│   │       └── synthesis_agent.py
│   ├── tools/
│   │   └── search.py    # Tavily wrapper
│   └── utils/
│       └── llm.py       # shared Groq instance (llama-3.3-70b-versatile)
└── frontend/            # Next.js app
    ├── app/
    ├── components/
    └── lib/
```

## Setup

### 1. Clone / open the project

```powershell
cd c:\Users\Rehan\Desktop\web-dev\synapse-ai
```

### 2. Create + activate a virtual environment

PowerShell (Windows):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

cmd (Windows):

```cmd
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the template and fill in real keys:

```powershell
copy .env.example .env
```

Then open `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- **Groq key** (free): https://console.groq.com/keys
- **Tavily key** (free tier available): https://app.tavily.com

### 5. Run the assistant (CLI)

```powershell
cd backend
python main.py
```

You'll get an interactive prompt:

```
============================================================
   Business Research Assistant
   Powered by Groq + Tavily + LangGraph
============================================================

You:
```

Type any business question; type `exit`, `quit`, `bye`, or `q` to leave.

## Running the full stack



Open **two terminals** from the **repository root** (`synapse-ai/`):

**Terminal 1 — FastAPI backend**

```powershell
cd backend
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Next.js frontend**

```powershell
cd frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

> The Next.js dev server proxies every `/api/*` request to `http://localhost:8000`
> so no CORS configuration is needed during development.

---

## Example queries to try

**Specific (will run straight through):**

- `What is Stripe's business model and recent funding?`
- `Tesla competitive position in the EV market 2026`
- `Analyse OpenAI's revenue and product strategy`
- `Shopify Q3 2025 earnings and key takeaways`
- `Give me a snapshot of Anthropic's product roadmap and investors`

**Vague (will trigger the human-feedback clarification loop):**

- `Tell me about some tech companies`  → you'll be asked to name a company
- `How is the market doing?`           → too broad, will ask for a sector
- `Some startup in fintech`            → no named entity, will ask

When clarification is requested, just type a more specific follow-up at the prompt and the graph resumes automatically.

## How multi-turn memory works

All queries within a session share one `thread_id`. The LangGraph `SqliteSaver` checkpointer persists `messages` to `data/conversations.db`, so conversation history survives server restarts and the synthesis agent sees prior exchanges as context. Per-query pipeline state (`clarity_status`, `research_attempts`, `validation_result`, …) is intentionally reset on every new query so stale verdicts don't leak forward.

## Customising

| Want to … | Edit |
| --- | --- |
| Use a different Groq model | Change `model=` in `get_llm()` inside `backend/utils/llm.py` (e.g. `mixtral-8x7b-32768`) |
| Allow more research retries | Change `_MAX_RESEARCH_ATTEMPTS` in `backend/graph/graph_builder.py` |
| Tighten / loosen the validator | Adjust the system prompt in `backend/graph/nodes/validator_agent.py` |
| Change Tavily depth | `search_tool = TavilySearch(max_results=...)` in `backend/tools/search.py` |
| Disable the data-quality disclaimer | Lower `_LOW_CONFIDENCE_THRESHOLD` in `backend/graph/nodes/synthesis_agent.py` |
| Switch to PostgreSQL persistence | Replace `SqliteSaver` with `PostgresSaver(conn=async_pg_pool)` in `backend/graph/graph_builder.py` |

## Troubleshooting

- **`GROQ_API_KEY not found in .env`** — your `.env` is missing or in the wrong folder. It must live at the **repository root** (next to `README.md`).
- **`TAVILY_API_KEY not found in .env`** — same as above; both keys must be set in the root `.env`.
- **`ModuleNotFoundError: langchain_tavily`** — re-run `pip install -r requirements.txt`. The non-deprecated Tavily tool lives in this package.
- **Graph hangs after clarification request** — make sure you're typing a clarification then pressing Enter at the `Your clarification:` prompt; the assistant is waiting on stdin.
- **`UnicodeEncodeError: 'charmap'`** — only triggers on Windows console with non-UTF8 code page; run `chcp 65001` in PowerShell before `python main.py` (from `backend/`) to force UTF-8.
- **Groq rate limit (429)** — the free tier allows ~30 requests/minute; wait a moment and retry. The error handler will print a helpful message rather than crashing.

## License

Personal/educational project.
