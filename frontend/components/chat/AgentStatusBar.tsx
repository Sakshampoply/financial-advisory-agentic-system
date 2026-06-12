"use client";
import { Spinner } from "@/components/ui/Spinner";

const NODE_LABELS: Record<string, string> = {
  guardrail_input: "Checking safety",
  intake: "Collecting your profile",
  document_intelligence: "Processing documents",
  profile_builder: "Building your profile",
  risk_assessment: "Analyzing portfolio risk",
  strategy: "Building allocation strategy",
  scoring: "Scoring your portfolio",
  advisor_copilot: "Generating your report",
  guardrail_output: "Finalizing response",
};

interface AgentStatusBarProps {
  activeNodes: string[];
  completedNodes: string[];
}

export function AgentStatusBar({ activeNodes, completedNodes }: AgentStatusBarProps) {
  const visibleNodes = [
    ...completedNodes.map((n) => ({ node: n, done: true })),
    ...activeNodes.map((n) => ({ node: n, done: false })),
  ];

  if (visibleNodes.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2 px-4 py-2 text-xs">
      {visibleNodes.map(({ node, done }) => (
        <span
          key={node}
          className="flex items-center gap-1.5"
          style={{ color: done ? "#22C55E" : "#E8A020" }}
        >
          {done ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : (
            <Spinner size={12} />
          )}
          {NODE_LABELS[node] ?? node}
        </span>
      ))}
    </div>
  );
}
