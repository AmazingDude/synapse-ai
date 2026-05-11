"use client";

import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { TriangleAlert } from "lucide-react";

interface QueryInputProps {
  onSubmit: (message: string) => void;
  isLoading: boolean;
  awaitingClarification: boolean;
  clarificationPrompt: string | null;
  disabled: boolean;
  showHints?: boolean;
}

/** Hint chips shown below the input when there are no messages yet. */
const HINT_CHIPS = ["Apple overview", "Tesla competitors", "Microsoft financials"] as const;

/**
 * Input area at the bottom of the left panel.
 * Shows an amber clarification banner when the agent is awaiting a follow-up.
 * Textarea auto-expands up to 4 lines; Enter submits, Shift+Enter inserts a newline.
 * Hint chips pre-fill and immediately submit common queries.
 */
export default function QueryInput({
  onSubmit,
  isLoading,
  awaitingClarification,
  clarificationPrompt,
  disabled,
  showHints = false,
}: QueryInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /** Derive placeholder from current mode. */
  const placeholder = awaitingClarification
    ? "Type your clarification…"
    : "Ask about any company…";

  /** Auto-resize the textarea up to four visible lines. */
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxHeight = 24 * 4 + 20; // 4 lines × ~24 px line-height + vertical padding
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, [value]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  /** Submit a value — optionally override the textarea content (used by chips). */
  const submit = (override?: string) => {
    const trimmed = (override ?? value).trim();
    if (!trimmed || isLoading || disabled) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <div className="flex flex-col gap-2 p-3">
      {/* ── Clarification banner ────────────────────────────────── */}
      {awaitingClarification && clarificationPrompt && (
        <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          <TriangleAlert className="w-4 h-4 text-amber-400 flex-none mt-px shrink-0" />
          <p className="text-xs text-amber-300 leading-relaxed">{clarificationPrompt}</p>
        </div>
      )}

      {/* ── Input row ───────────────────────────────────────────── */}
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none rounded-lg bg-[#1a1a1a] border border-neutral-700 text-[#f5f5f5] text-sm px-3 py-2.5 placeholder-neutral-600 focus:outline-none focus:border-indigo-500 transition-colors min-h-[44px] leading-6"
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isLoading}
        />

        {/* Arrow / spinner submit button */}
        <button
          onClick={() => submit()}
          disabled={isLoading || !value.trim() || disabled}
          className="flex-none w-[44px] h-[44px] flex items-center justify-center rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          aria-label="Submit"
        >
          {isLoading ? (
            <span className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          )}
        </button>
      </div>

      {/* ── Hint chips ──────────────────────────────────────────── */}
      {showHints && !isLoading && !awaitingClarification && (
        <div className="flex flex-wrap gap-1.5">
          {HINT_CHIPS.map((chip) => (
            <button
              key={chip}
              onClick={() => submit(chip)}
              className="px-2.5 py-1 rounded-full bg-neutral-800 border border-neutral-700 text-xs text-neutral-500 hover:text-neutral-200 hover:border-indigo-500/50 transition-colors cursor-pointer"
            >
              {chip}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
