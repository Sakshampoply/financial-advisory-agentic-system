"use client";
import { MessageList } from "./MessageList";
import { AgentStatusBar } from "./AgentStatusBar";
import { InputBar } from "./InputBar";
import type { Message } from "@/lib/types";

interface ChatMainProps {
  messages: Message[];
  activeNodes: string[];
  completedNodes: string[];
  isStreaming: boolean;
  uploading: boolean;
  sessionId: string | null;
  onSend: (content: string) => void;
  onFile: (file: File) => void;
}

export function ChatMain({
  messages,
  activeNodes,
  completedNodes,
  isStreaming,
  uploading,
  sessionId,
  onSend,
  onFile,
}: ChatMainProps) {
  return (
    <main className="flex-1 flex flex-col overflow-hidden min-w-0">
      <MessageList messages={messages} isStreaming={isStreaming} />
      {(activeNodes.length > 0 || (isStreaming && completedNodes.length > 0)) && (
        <AgentStatusBar activeNodes={activeNodes} completedNodes={completedNodes} />
      )}
      <InputBar
        onSend={onSend}
        onFile={onFile}
        isStreaming={isStreaming}
        uploading={uploading}
        disabled={!sessionId}
      />
    </main>
  );
}
