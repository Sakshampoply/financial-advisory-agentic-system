"use client";
import { MarkdownRenderer } from "./MarkdownRenderer";
import type { Message } from "@/lib/types";

const AGENT_LABELS: Record<string, string> = {
  intake: "Profile Collection",
  advisor_copilot: "Financial Advisor",
  risk_assessment: "Risk Analysis",
  strategy: "Strategy",
  scoring: "Portfolio Scoring",
  document_intelligence: "Document Processing",
  profile_builder: "Profile Builder",
  guardrail_input: "Safety Check",
  guardrail_output: "Safety Check",
};

export function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="message-enter flex justify-end px-4 py-2">
        <div
          className="max-w-[70%] rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm"
          style={{ background: "#1E2A3A", color: "#F0F4F8" }}
        >
          {message.content}
        </div>
      </div>
    );
  }

  const agentLabel = message.agent ? AGENT_LABELS[message.agent] ?? message.agent : null;

  return (
    <div className="message-enter px-4 py-2">
      <div
        className="rounded-xl rounded-tl-sm p-4"
        style={{ background: "#0F1520", border: "1px solid #1E2A3A" }}
      >
        {agentLabel && (
          <div className="text-xs font-semibold mb-2" style={{ color: "#E8A020" }}>
            {agentLabel}
          </div>
        )}
        {message.content ? (
          <MarkdownRenderer content={message.content} streaming={message.streaming} />
        ) : (
          <div className="flex items-center gap-2 text-sm" style={{ color: "#6B7E96" }}>
            <span className="typing-cursor" />
          </div>
        )}
      </div>
    </div>
  );
}
