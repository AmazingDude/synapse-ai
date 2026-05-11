/**
 * API client for the Business Research Assistant backend.
 * All calls use relative URLs so Next.js rewrites proxy them to
 * http://localhost:8000 in development without CORS issues.
 */


// ─── Handler interface ────────────────────────────────────────────

/** Callbacks invoked as SSE frames arrive from the backend. */
export interface StreamHandlers {
  /** A LangGraph node just started executing. */
  onAgentStart: (agent: string, label: string) => void;
  /** Graph paused — user must provide more detail before research can continue. */
  onClarificationNeeded: (message: string) => void;
  /** Final markdown report is ready. */
  onReport: (content: string) => void;
  /** A recoverable or fatal error occurred during the run. */
  onError: (message: string) => void;
  /** Stream is fully consumed — loading state can be cleared. */
  onDone: () => void;
}

// ─── Internal SSE types ───────────────────────────────────────────

/** Shape of every JSON object emitted by the backend as an SSE frame. */
interface SSEEvent {
  type: "agent_start" | "clarification_needed" | "report" | "error" | "done";
  /** Present on agent_start events. */
  agent?: string;
  /** Human-readable label for the running agent node. */
  label?: string;
  /** Clarification prompt text or error description. */
  message?: string;
  /** Markdown report body. */
  content?: string;
}

// ─── SSE stream parser ────────────────────────────────────────────

/**
 * Reads a `text/event-stream` response body and dispatches every JSON event
 * to the appropriate handler.
 *
 * The SSE format from the backend is:
 *   `data: <json>\n\n`
 *
 * Incomplete frames (split across TCP chunks) are buffered until a `\n\n`
 * delimiter is received.
 */
async function consumeStream(
  response: Response,
  handlers: StreamHandlers,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) {
    handlers.onError("Response body is empty — is the backend running?");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Each SSE event is terminated by a blank line (\n\n).
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? ""; // keep any incomplete trailing block

      for (const block of blocks) {
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;

          const raw = line.slice(6).trim();
          if (!raw) continue;

          try {
            const event = JSON.parse(raw) as SSEEvent;
            dispatch(event, handlers);
          } catch {
            // Silently skip malformed JSON frames.
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Route one parsed SSE event to the correct handler. */
function dispatch(event: SSEEvent, handlers: StreamHandlers): void {
  switch (event.type) {
    case "agent_start":
      handlers.onAgentStart(event.agent ?? "", event.label ?? "");
      break;
    case "clarification_needed":
      handlers.onClarificationNeeded(event.message ?? "");
      break;
    case "report":
      handlers.onReport(event.content ?? "");
      break;
    case "error":
      handlers.onError(event.message ?? "Unknown error");
      break;
    case "done":
      handlers.onDone();
      break;
  }
}

// ─── Public API ───────────────────────────────────────────────────

/**
 * POST /api/research — start a new research query.
 *
 * Streams progress events back to the caller via `handlers` as the
 * LangGraph pipeline executes. If the graph requires clarification the
 * stream ends with a `clarification_needed` event; the caller should
 * then prompt the user and call `streamClarification`.
 *
 * @param message   The user's research question.
 * @param threadId  Unique session UUID; used by the backend's MemorySaver.
 * @param handlers  Callbacks for each SSE event type.
 */
export async function streamResearch(
  message: string,
  threadId: string,
  handlers: StreamHandlers,
): Promise<void> {
  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      handlers.onError(`Server error ${response.status}: ${text || response.statusText}`);
      handlers.onDone();
      return;
    }

    await consumeStream(response, handlers);
  } catch (err) {
    handlers.onError(
      err instanceof Error
        ? `Network error: ${err.message}`
        : "Network error — is the backend running on port 8000?",
    );
    handlers.onDone();
  }
}

/**
 * POST /api/clarify — resume a graph paused for human clarification.
 *
 * The `threadId` must match the one used in the preceding `streamResearch`
 * call so the backend can restore the interrupted checkpoint.
 *
 * @param clarification  The user's clarification answer.
 * @param threadId       Session UUID matching the interrupted research call.
 * @param handlers       Same callbacks as `streamResearch`.
 */
export async function streamClarification(
  clarification: string,
  threadId: string,
  handlers: StreamHandlers,
): Promise<void> {
  try {
    const response = await fetch("/api/clarify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clarification, thread_id: threadId }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      handlers.onError(`Server error ${response.status}: ${text || response.statusText}`);
      handlers.onDone();
      return;
    }

    await consumeStream(response, handlers);
  } catch (err) {
    handlers.onError(
      err instanceof Error
        ? `Network error: ${err.message}`
        : "Network error — is the backend running on port 8000?",
    );
    handlers.onDone();
  }
}

/**
 * GET /api/health — check whether the FastAPI backend is reachable.
 *
 * @returns `true` when the backend responds with `{ "status": "ok" }`.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch("/api/health", { method: "GET" });
    if (!response.ok) return false;
    const data = (await response.json()) as { status?: string };
    return data.status === "ok";
  } catch {
    return false;
  }
}
