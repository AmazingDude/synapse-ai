# Business Research Assistant

A multi-agent business research CLI built with **LangGraph**, **Gemini** (Google AI Studio), and **Tavily**. Type any business question; the system clarifies it if vague, runs targeted web searches, validates the findings, then writes you a structured report.

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

```
business-research-assistant/
├── .env                # your real API keys (gitignored)
├── .env.example        # template
├── main.py             # interactive CLI runner
├── requirements.txt
├── README.md
├── graph/
│   ├── __init__.py     # empty by design (avoids eager LLM init)
│   ├── state.py        # GraphState (TypedDict + add_messages reducer)
│   ├── graph_builder.py
│   └── nodes/
│       ├── __init__.py
│       ├── clarity_agent.py
│       ├── research_agent.py
│       ├── validator_agent.py
│       └── synthesis_agent.py
├── tools/
│   ├── __init__.py
│   └── search.py       # Tavily wrapper (search_tool + helpers)
└── utils/
    ├── __init__.py
    └── llm.py          # shared Gemini llm instance
```

## Setup

### 1. Clone / open the project

```powershell
cd c:\Users\Rehan\Desktop\web-dev\synapse-ai\business-research-assistant
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
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- **Gemini key** (free): https://aistudio.google.com/apikey
- **Tavily key** (free tier available): https://app.tavily.com

### 5. Run the assistant

```powershell
python main.py
```

You'll get an interactive prompt:

```
============================================================
   Business Research Assistant
   Powered by Gemini + Tavily + LangGraph
============================================================

You:
```

Type any business question; type `exit`, `quit`, `bye`, or `q` to leave.

## Running the full stack



Open **two terminals** from the `business-research-assistant/` directory:

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

All queries within a session share one `thread_id`. The LangGraph `MemorySaver` checkpointer accumulates `messages` across turns, so the synthesis agent sees the last few conversational exchanges as context. Per-query pipeline state (`clarity_status`, `research_attempts`, `validation_result`, …) is intentionally reset on every new query so stale verdicts don't leak forward.

## Customising

| Want to … | Edit |
| --- | --- |
| Use a different Gemini model | Set `GEMINI_MODEL` in `.env` (e.g. `gemini-2.5-pro`) |
| Allow more research retries | Change `_MAX_RESEARCH_ATTEMPTS` in `graph/graph_builder.py` |
| Tighten / loosen the validator | Adjust `_SCORE_THRESHOLD_LOW` and `_SCORE_THRESHOLD_HIGH` in `graph/nodes/validator_agent.py` |
| Change Tavily depth | `search_tool = TavilySearch(max_results=...)` in `tools/search.py` |
| Disable the data-quality disclaimer | Lower `_LOW_CONFIDENCE_THRESHOLD` in `graph/nodes/synthesis_agent.py` |
| Persist state to disk | Replace `MemorySaver` with `SqliteSaver` or `PostgresSaver` in `graph/graph_builder.py` |

## Troubleshooting

- **`GOOGLE_API_KEY is not set` / `TAVILY_API_KEY is not set`** — your `.env` is missing or in the wrong folder. It must live next to `main.py`.
- **`ModuleNotFoundError: langchain_tavily`** — re-run `pip install -r requirements.txt`. The non-deprecated Tavily tool lives in this package.
- **Graph hangs after clarification request** — make sure you're typing a clarification then pressing Enter at the `Your clarification:` prompt; the assistant is waiting on stdin.
- **`UnicodeEncodeError: 'charmap'`** — only triggers on Windows console with non-UTF8 code page; run `chcp 65001` in PowerShell before `python main.py` to force UTF-8.

## License

Personal/educational project.
