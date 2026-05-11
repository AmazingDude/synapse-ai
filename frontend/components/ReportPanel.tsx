"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import { ChevronRight } from "lucide-react";

interface ReportPanelProps {
  report: string | null;
  isLoading: boolean;
  confidence: number | null;
  onChipClick: (query: string) => void;
}

/** Example chips shown on the empty-state placeholder. */
const EXAMPLE_CHIPS = [
  "Tell me about Apple",
  "Tesla's latest financials",
  "Microsoft vs Google cloud",
] as const;

/** Skeleton widths for the loading shimmer blocks. */
const SKELETON_WIDTHS = ["w-3/4", "w-full", "w-5/6", "w-4/5", "w-1/2", "w-full", "w-2/3"] as const;

/** Custom renderers that apply the dark design system to every markdown element. */
const markdownComponents: Components = {
  h2: ({ children }) => (
    <h2 className="text-xl font-semibold text-indigo-400 border-b border-neutral-800 pb-2 mt-7 mb-3 first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-base font-semibold text-neutral-200 mt-5 mb-2">
      {children}
    </h3>
  ),
  p: ({ children }) => (
    <p className="text-neutral-300 leading-relaxed mb-4">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="mb-4 space-y-1.5">{children}</ul>
  ),
  li: ({ children }) => (
    <li className="flex gap-1.5 text-neutral-300">
      <ChevronRight className="w-3.5 h-3.5 text-indigo-400 mt-[2px] flex-none shrink-0" />
      <span>{children}</span>
    </li>
  ),
  strong: ({ children }) => (
    <strong className="text-white font-semibold">{children}</strong>
  ),
  blockquote: ({ children }) => (
    <blockquote className="bg-amber-500/10 border-l-4 border-amber-500/50 text-amber-300 rounded-r px-4 py-2 my-4">
      {children}
    </blockquote>
  ),
};

/** Small confidence badge shown at the top of a loaded report. */
function ConfidenceBar({ confidence }: { confidence: number | null }) {
  if (confidence === null) return null;

  const { dotClass, label } =
    confidence >= 7
      ? { dotClass: "bg-green-500", label: "High confidence" }
      : confidence >= 5
        ? { dotClass: "bg-amber-500", label: "Medium confidence" }
        : { dotClass: "bg-red-500", label: "Low confidence" };

  return (
    <div className="flex items-center gap-2 text-xs text-neutral-400">
      <span className={`w-2 h-2 rounded-full ${dotClass}`} />
      <span>{label}</span>
    </div>
  );
}

/**
 * Right-panel report display.
 * Three states: empty placeholder → shimmer skeleton while loading → rendered markdown report.
 */
export default function ReportPanel({
  report,
  isLoading,
  confidence,
  onChipClick,
}: ReportPanelProps) {

  /* ── Skeleton ─────────────────────────────────────────────────── */
  if (isLoading && !report) {
    return (
      <div className="p-8 max-w-3xl mx-auto">
        <div className="flex flex-col gap-3">
          {SKELETON_WIDTHS.map((w, i) => (
            <div
              key={i}
              className={`skeleton-shimmer h-4 rounded-md ${w} ${i === 4 ? "mt-4" : ""}`}
            />
          ))}
        </div>
      </div>
    );
  }

  /* ── Empty placeholder ────────────────────────────────────────── */
  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-5 px-8 text-center">
        {/* Magnifying glass icon */}
        <div className="w-14 h-14 rounded-2xl bg-neutral-800/60 flex items-center justify-center">
          <svg
            className="w-7 h-7 text-neutral-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
            />
          </svg>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-neutral-200 mb-1.5">
            Ask anything about a business
          </h2>
          <p className="text-sm text-neutral-500 max-w-sm leading-relaxed">
            Get structured research reports powered by live web data
          </p>
        </div>

        {/* Example chips */}
        <div className="flex flex-wrap gap-2 justify-center">
          {EXAMPLE_CHIPS.map((chip) => (
            <button
              key={chip}
              onClick={() => onChipClick(chip)}
              className="px-3 py-1.5 rounded-full bg-neutral-800 border border-neutral-700 text-xs text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200 hover:border-indigo-500/50 transition-colors cursor-pointer"
            >
              {chip}
            </button>
          ))}
        </div>
      </div>
    );
  }

  /* ── Report ───────────────────────────────────────────────────── */
  // `key={report}` forces React to remount the fade-in div whenever the
  // report content changes, restarting the CSS animation naturally without
  // needing an effect-driven opacity toggle.
  return (
    <div key={report} className="p-8 max-w-3xl mx-auto report-fade-in">
      <div className="flex items-center mb-6 pb-4 border-b border-neutral-800">
        <ConfidenceBar confidence={confidence} />
      </div>

      <div className="text-sm">
        <ReactMarkdown components={markdownComponents} remarkPlugins={[remarkGfm]}>
          {report}
        </ReactMarkdown>
      </div>
    </div>
  );
}
