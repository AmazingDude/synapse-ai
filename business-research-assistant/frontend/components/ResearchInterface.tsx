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
   * Replace the last system message in the messages array with new content.
   * If the last message is not a system message, appends a new one instead.
   * This makes agent-progress labels animate in-place rather than stacking.
   */
  const updateOrAppendSystemMessage = (content: string, variant?: SystemVariant) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "system") {
        return [...prev.slice(0, -1), { ...last, content, variant }];
      }
      return [...prev, { id: crypto.randomUUID(), role: "system", content, variant }];
    });
  };

  /**
   * Shared SSE handler factory — avoids duplicating handler logic between
   * handleSubmit and handleClarification.
   */
  const buildHandlers = (): StreamHandlers => ({
    onAgentStart: (agent, _label) => {
      const agentLabels: Record<string, string> = {
        clarity_agent:   "🔍 Evaluating query clarity...",
        research_agent:  "📊 Searching for company data...",
        validator_agent: "✅ Validating research quality...",
        synthesis_agent: "📝 Writing your report...",
      };
      const label = agentLabels[agent] ?? _label;
      setAgentStatus(label);
      // Always replace the last message unconditionally — an initial system
      // pill is guaranteed to exist (added in handleSubmit / handleClarification
      // before the stream starts). Conditional replace caused duplicates when
      // the state snapshot inside the closure hadn't refreshed yet.
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { id: crypto.randomUUID(), role: "system" as MessageRole, content: label },
      ]);
    },
    onClarificationNeeded: (msg) => {
      setAwaitingClarification(true);
      setClarificationPrompt(msg);
      setIsLoading(false);
      setAgentStatus(null);
    },
    onReport: (content) => {
      setCurrentReport(content);
      setMessages((prev) => {
        const filtered = prev.filter((m) => !(m.role === "system" && m.content === "✅ Report ready"));
        return [...filtered, { id: crypto.randomUUID(), role: "system", content: "✅ Report ready", variant: "success" as SystemVariant }];
      });
    },
    onError: (msg) => {
      addMessage("system", msg, "error");

      setIsLoading(false);
      setAgentStatus(null);
    },
    onDone: () => {
      setIsLoading(false);
      setAgentStatus(null);
      updateOrAppendSystemMessage("✅ Done");
      setTimeout(() => {
        setMessages((prev) => prev.filter((m) => !(m.role === "system" && m.content === "✅ Done")));
      }, 2000);
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
    setAgentStatus("🔍 Evaluating query clarity...");
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "system", content: "🔍 Evaluating query clarity..." },
    ]);

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
    setAgentStatus("🔍 Evaluating query clarity...");
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "system", content: "🔍 Evaluating query clarity..." },
    ]);

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
