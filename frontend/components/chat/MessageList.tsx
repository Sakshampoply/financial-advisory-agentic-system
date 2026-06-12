"use client";
import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import type { Message } from "@/lib/types";

interface MessageListProps {
  messages: Message[];
  isStreaming: boolean;
}

export function MessageList({ messages, isStreaming }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4 px-6 text-center">
        <div
          className="w-14 h-14 rounded-2xl flex items-center justify-center"
          style={{ background: "#0F1520", border: "1px solid #1E2A3A" }}
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#E8A020" strokeWidth="1.5">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </div>
        <div>
          <p className="font-semibold text-base" style={{ color: "#F0F4F8" }}>
            Financial Advisor AI
          </p>
          <p className="text-sm mt-1" style={{ color: "#6B7E96" }}>
            Ask me about your investments, portfolio risk, or market strategy.
            <br />
            You can also upload portfolio documents to get a personalized analysis.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 mt-2 w-full max-w-md">
          {[
            "What is my portfolio risk?",
            "Create an investment plan for me",
            "What is an ETF?",
            "How do I diversify my holdings?",
          ].map((prompt) => (
            <button
              key={prompt}
              className="text-left px-3 py-2.5 rounded-lg text-xs transition-colors cursor-pointer"
              style={{
                background: "#0F1520",
                border: "1px solid #1E2A3A",
                color: "#6B7E96",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLButtonElement).style.borderColor = "#E8A020";
                (e.currentTarget as HTMLButtonElement).style.color = "#F0F4F8";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.borderColor = "#1E2A3A";
                (e.currentTarget as HTMLButtonElement).style.color = "#6B7E96";
              }}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto py-2">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
