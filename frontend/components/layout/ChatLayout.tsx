"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { TopBar } from "./TopBar";
import { Sidebar } from "./Sidebar";
import { ChatMain } from "@/components/chat/ChatMain";
import { MetricsPanel } from "@/components/metrics/MetricsPanel";
import { useSession } from "@/hooks/useSession";
import { useChat } from "@/hooks/useChat";
import { useDocuments } from "@/hooks/useDocuments";
import type { Session } from "@/lib/types";

export function ChatLayout() {
  const { sessions, activeSession, loading, newSession, switchSession } = useSession();
  const { messages, activeNodes, completedNodes, isStreaming, analysisData, send, loadHistory, reset } =
    useChat(activeSession?.id ?? null);
  const { documents, uploading, upload, loadDocuments } = useDocuments(activeSession?.id ?? null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // One-time initial load when the session list finishes loading
  const initialLoadRef = useRef(false);
  useEffect(() => {
    if (!loading && !initialLoadRef.current) {
      initialLoadRef.current = true;
      if (activeSession?.id) {
        loadHistory(activeSession.id);
        loadDocuments(activeSession.id);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  // Reset chat state when no session is active
  useEffect(() => {
    if (!loading && !activeSession?.id) {
      reset();
    }
  }, [activeSession?.id, loading, reset]);

  const handleNewChat = useCallback(async () => {
    const session = await newSession();
    reset();
    setSidebarOpen(false);
    return session;
  }, [newSession, reset]);

  const handleSelectSession = useCallback(
    (session: Session) => {
      switchSession(session);
      reset();
      loadHistory(session.id);
      loadDocuments(session.id);
      setSidebarOpen(false);
    },
    [switchSession, reset, loadHistory, loadDocuments]
  );

  const handleFile = useCallback(
    async (file: File) => {
      let sid = activeSession?.id;
      if (!sid) {
        const session = await newSession();
        sid = session.id;
      }
      await upload(file, sid);
    },
    [activeSession?.id, newSession, upload]
  );

  const handleSend = useCallback(
    async (content: string) => {
      let sid = activeSession?.id;
      if (!sid) {
        const session = await newSession();
        sid = session.id;
      }
      send(content, sid);
    },
    [activeSession?.id, newSession, send]
  );

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center" style={{ background: "#080C14" }}>
        <div className="flex flex-col items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: "#E8A020" }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#080C14" strokeWidth="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <p className="text-sm" style={{ color: "#6B7E96" }}>
            Loading…
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: "#080C14" }}>
      <TopBar
        onNewChat={handleNewChat}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
      />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSession?.id ?? null}
          onSelectSession={handleSelectSession}
          documents={documents}
          open={sidebarOpen}
        />
        <ChatMain
          messages={messages}
          activeNodes={activeNodes}
          completedNodes={completedNodes}
          isStreaming={isStreaming}
          uploading={uploading}
          sessionId={activeSession?.id ?? null}
          onSend={handleSend}
          onFile={handleFile}
        />
        <MetricsPanel data={analysisData} />
      </div>
    </div>
  );
}
