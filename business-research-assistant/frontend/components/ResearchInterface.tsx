"use client";

import { useState, useEffect } from "react";
import ChatPanel from "./ChatPanel";
import ReportPanel from "./ReportPanel";
import QueryInput from "./QueryInput";
import {
  streamResearch,
  streamClarification,
  checkHealth,
  type StreamHandlers,
} from "../lib/api";

/* ─── Type definitions ─────────────────────────────────────────── */

export type MessageRole = "user" | "assistant" | "system";

/** Visual variant applied to system messages; drives icon selection in ChatPanel. */
export type SystemVariant = "success" | "error" | "warning" | "info";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  variant?: SystemVariant;
}

export interface ResearchState {
  messages: Message[];
  currentReport: string | null;
  isLoading: boolean;
  agentStatus: string | null;
  threadId: string;
  awaitingClarification: boolean;
  clarificationPrompt: string | null;
  confidence: number | null;
}

/* ─── Component ────────────────────────────────────────────────── */

/**
 * Root shell component for the research assistant.
 * Manages all application state and renders the two-column layout:
 *   LEFT  — chat history + query input
 *   RIGHT — live report display
 */
export default function ResearchInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentReport, setCurrentReport] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [agentStatus, setAgentStatus] = useState<string | null>(null);
  /** Stable session ID — initialised lazily so we don't trigger a cascading render. */
  const [threadId] = useState<string>(() => crypto.randomUUID());
  const [awaitingClarification, setAwaitingClarification] = useState(false);
  const [clarificationPrompt, setClarificationPrompt] = useState<
    string | null
  >(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  /** Verify backend connectivity on mount; show a warning if the server is unreachable. */
  useEffect(() => {
    checkHealth().then((ok) => {
      if (!ok) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "system",
            variant: "warning",
            content:
              "Cannot connect to backend. Make sure the server is running on port 8000.",
          },
        ]);
      }
    });
  }, []);

  /**
   * Append a new message to the conversation history.
   * @param role    - Who sent the message.
   * @param content - Message body text.
   * @param variant - Optional visual variant for system messages (drives icon in ChatPanel).
   */
  const addMessage = (role: MessageRole, content: string, variant?: SystemVariant) => {
    const msg: Message = {
      id: crypto.randomUUID(),
      role,
      content,
      variant,
    };
    setMessages((prev) => [...prev, msg]);
    return msg;
  };

  /**
   * Shared SSE handler factory — avoids duplicating handler logic between
   * handleSubmit and handleClarification.
   */
  const buildHandlers = (): StreamHandlers => ({
    onAgentStart: (_agent, label) => setAgentStatus(label),
    onClarificationNeeded: (msg) => {
      setAwaitingClarification(true);
      setClarificationPrompt(msg);
      setIsLoading(false);
      setAgentStatus(null);
    },
    onReport: (content) => {
      setCurrentReport(content);
      addMessage("system", "Report ready", "success");
    },
    onError: (msg) => {
      addMessage("system", msg, "error");

      setIsLoading(false);
      setAgentStatus(null);
    },
    onDone: () => {
      setIsLoading(false);
      setAgentStatus(null);
    },
  });

  /**
   * Handle a query submitted by the user.
   * Clears the previous report, adds the user message, then streams the
   * research pipeline via SSE.
   * @param query - The research question entered by the user.
   */
  const handleSubmit = async (query: string) => {
    if (!query.trim() || isLoading) return;

    addMessage("user", query);
    setIsLoading(true);
    setCurrentReport(null);
    setConfidence(null);
    setAgentStatus("Initialising research…");

    await streamResearch(query, threadId, buildHandlers());
  };

  /**
   * Handle a clarification answer submitted by the user.
   * Resets the clarification state, echoes the answer into the chat, then
   * resumes the interrupted graph via /api/clarify.
   * @param answer - The user's response to the clarification prompt.
   */
  const handleClarification = async (answer: string) => {
    if (!answer.trim()) return;

    setAwaitingClarification(false);
    setClarificationPrompt(null);
    addMessage("user", answer);
    setIsLoading(true);
    setAgentStatus("Resuming research…");

    await streamClarification(answer, threadId, buildHandlers());
  };

  return (
    <div className="flex h-full bg-[#0f0f0f] text-[#f5f5f5] overflow-hidden">
      {/* ── LEFT PANEL ──────────────────────────────────────────── */}
      <aside className="w-[380px] flex-none flex flex-col border-r border-[#2a2a2a]">
        {/* Header */}
        <header className="flex-none px-6 py-5 border-b border-[#2a2a2a]">
          <h1 className="text-xl font-bold tracking-tight header-gradient">
            Synapse
          </h1>
          <p className="text-xs text-[#a3a3a3] mt-0.5">Research Assistant</p>
        </header>

        {/* Chat history */}
        <div className="flex-1 overflow-y-auto">
          <ChatPanel
            messages={messages}
            isLoading={isLoading}
            agentStatus={agentStatus}
          />
        </div>

        {/* Input area */}
        <div className="flex-none border-t border-[#2a2a2a]">
          <QueryInput
            onSubmit={awaitingClarification ? handleClarification : handleSubmit}
            isLoading={isLoading}
            awaitingClarification={awaitingClarification}
            clarificationPrompt={clarificationPrompt}
            disabled={isLoading}
            showHints={messages.length === 0}
          />
        </div>
      </aside>

      {/* ── RIGHT PANEL ─────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <ReportPanel
          report={currentReport}
          isLoading={isLoading}
          confidence={confidence}
          onChipClick={handleSubmit}
        />
      </main>
    </div>
  );
}
