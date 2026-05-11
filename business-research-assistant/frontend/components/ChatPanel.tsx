"use client";

import { useEffect, useRef } from "react";
import { TriangleAlert, CircleCheck, CircleX, Info } from "lucide-react";
import type { Message, SystemVariant } from "./ResearchInterface";

/** Maps a system-message variant to its Lucide icon + colour classes. */
const VARIANT_META: Record<
  NonNullable<SystemVariant>,
  { Icon: React.ElementType; iconClass: string; textClass: string }
> = {
  success: { Icon: CircleCheck,   iconClass: "text-green-500",  textClass: "text-green-400"  },
  error:   { Icon: CircleX,       iconClass: "text-red-500",    textClass: "text-red-400"    },
  warning: { Icon: TriangleAlert, iconClass: "text-amber-500",  textClass: "text-amber-400"  },
  info:    { Icon: Info,          iconClass: "text-indigo-400", textClass: "text-neutral-400" },
};

/** A centered system status row with an icon keyed to the message variant. */
function SystemMessage({ // CHANGED: now accepts isLive to show pulsing dot
  content,
  variant,
  isLive = false, // CHANGED: true when isLoading and this is the last message
}: {
  content: string;
  variant?: SystemVariant;
  isLive?: boolean; // CHANGED
}) {
  const meta = variant ? VARIANT_META[variant] : null;
  const { Icon, iconClass, textClass } = meta ?? {
    Icon: null,
    iconClass: "",
    textClass: "text-neutral-400", // CHANGED: slightly brighter for live progress pills
  };

  return (
    <div className="flex items-center justify-center gap-1.5 py-0.5"> {/* CHANGED: centered pill */}
      {isLive && ( // CHANGED: pulsing dot shown only when this is the active live message
        <span className="animate-pulse w-2 h-2 rounded-full bg-indigo-400 inline-block mr-2" /> // CHANGED
      )}
      {!isLive && Icon && <Icon className={`w-3.5 h-3.5 flex-none ${iconClass}`} />} {/* CHANGED: hide icon when live dot shown */}
      <p className={`text-xs ${textClass}`}>{content}</p>
    </div>
  );
}

interface ChatPanelProps {
  messages: Message[];
  agentStatus: string | null;
  isLoading: boolean;
}

/**
 * Scrollable conversation history.
 * User messages: right-aligned indigo bubbles.
 * System messages: centered status lines with optional pulsing dot.
 * Assistant messages: left-aligned neutral bubbles.
 * Typing indicator appears at the bottom while the agent is running.
 */
export default function ChatPanel({
  messages,
  agentStatus,
  isLoading,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  /** Scroll to the bottom whenever the message list or loading state changes. */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col gap-3 p-4 min-h-full">
      {/* ── Empty state ─────────────────────────────────────────── */}
      {messages.length === 0 && !isLoading && (
        <p className="text-xs text-neutral-600 text-center mt-10 select-none">
          Start a research query to begin.
        </p>
      )}

      {/* ── Message list ────────────────────────────────────────── */}
      {messages.map((msg, idx) => { // CHANGED: track index to detect last message
        if (msg.role === "user") {
          return (
            <div key={msg.id} className="flex flex-col items-end gap-1">
              <span className="text-[10px] font-medium text-indigo-400 mr-1">
                You
              </span>
              <div className="bg-indigo-600/20 border border-indigo-500/30 rounded-2xl rounded-tr-sm px-3 py-2 max-w-[90%]">
                <p className="text-sm text-[#f5f5f5] leading-relaxed whitespace-pre-wrap">
                  {msg.content}
                </p>
              </div>
            </div>
          );
        }

        if (msg.role === "system") {
          const isLastMessage = idx === messages.length - 1; // CHANGED
          const isLive = isLoading && isLastMessage; // CHANGED: pulse only on the active last pill
          return (
            <SystemMessage key={msg.id} content={msg.content} variant={msg.variant} isLive={isLive} /> // CHANGED
          );
        }

        // assistant
        return (
          <div key={msg.id} className="flex flex-col items-start gap-1">
            <span className="text-[10px] font-medium text-neutral-500 ml-1">
              Synapse
            </span>
            <div className="bg-neutral-800/60 border border-neutral-700/50 rounded-2xl rounded-tl-sm px-3 py-2 max-w-[90%]">
              <p className="text-sm text-neutral-200 leading-relaxed whitespace-pre-wrap">
                {msg.content}
              </p>
            </div>
          </div>
        );
      })}

      {/* ── Typing indicator ────────────────────────────────────── */}
      {isLoading && agentStatus && (
        <div className="flex items-center gap-2.5 py-1">
          <div className="flex gap-1">
            {[0, 150, 300].map((delay) => (
              <span
                key={delay}
                className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse"
                style={{ animationDelay: `${delay}ms` }}
              />
            ))}
          </div>
          <p className="text-xs text-neutral-500 italic">{agentStatus}</p>
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
}
