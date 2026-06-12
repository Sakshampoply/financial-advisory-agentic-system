"use client";
import { useState, useCallback, useRef } from "react";
import { sendMessage, getSessionMessages } from "@/lib/api";
import { streamSSE } from "@/lib/sse";
import type { Message, AnalysisData } from "@/lib/types";

let idCounter = 0;
function uid() {
  return `msg-${Date.now()}-${++idCounter}`;
}

export function useChat(sessionId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeNodes, setActiveNodes] = useState<string[]>([]);
  const [completedNodes, setCompletedNodes] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisData>({});
  const abortRef = useRef<(() => void) | null>(null);

  const loadHistory = useCallback(async (sid: string) => {
    const history = await getSessionMessages(sid);
    const msgs: Message[] = history.map((m) => ({
      id: uid(),
      role: m.role as "user" | "assistant",
      content: m.content,
      agent: m.agent,
      timestamp: Date.now(),
    }));
    setMessages(msgs);
    setActiveNodes([]);
    setCompletedNodes([]);
    setAnalysisData({});
  }, []);

  const send = useCallback(
    async (content: string, overrideSessionId?: string) => {
      const sid = overrideSessionId ?? sessionId;
      if (!sid || isStreaming) return;

      const userMsg: Message = { id: uid(), role: "user", content, timestamp: Date.now() };
      const assistantMsgId = uid();
      const assistantMsg: Message = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        streaming: true,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);
      setActiveNodes([]);
      setCompletedNodes([]);

      // Mutable ref tracks the current placeholder ID for the rolling-placeholder pattern
      const currentIdRef = { current: assistantMsgId };

      try {
        const stream = sendMessage(sid, content);
        abortRef.current = () => stream.cancel?.();

        for await (const evt of streamSSE(stream)) {
          if (evt.type === "node_start") {
            const node = evt.data.node as string;
            setActiveNodes((prev) => [...prev.filter((n) => n !== node), node]);
          } else if (evt.type === "node_complete") {
            const node = evt.data.node as string;
            setActiveNodes((prev) => prev.filter((n) => n !== node));
            setCompletedNodes((prev) => [...prev, node]);
          } else if (evt.type === "token") {
            const token = evt.data.content as string;
            const cid = currentIdRef.current;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === cid ? { ...m, content: m.content + token } : m
              )
            );
          } else if (evt.type === "message") {
            const msgContent = evt.data.content as string;
            const agent = evt.data.agent as string | undefined;
            const cid = currentIdRef.current;
            // Finalize current placeholder with the authoritative committed message
            setMessages((prev) =>
              prev.map((m) =>
                m.id === cid ? { ...m, content: msgContent, agent, streaming: false } : m
              )
            );
            // Create next placeholder for any subsequent agent messages
            const nextId = uid();
            currentIdRef.current = nextId;
            setMessages((prev) => [
              ...prev,
              { id: nextId, role: "assistant", content: "", streaming: true, timestamp: Date.now() },
            ]);
          } else if (evt.type === "state") {
            if (evt.data.risk_metrics) {
              setAnalysisData((prev) => ({
                ...prev,
                risk_metrics: evt.data.risk_metrics as AnalysisData["risk_metrics"],
              }));
            }
            if (evt.data.allocation_result) {
              setAnalysisData((prev) => ({
                ...prev,
                allocation_result: evt.data.allocation_result as AnalysisData["allocation_result"],
              }));
            }
            if (evt.data.scoring_result) {
              setAnalysisData((prev) => ({
                ...prev,
                scoring_result: evt.data.scoring_result as AnalysisData["scoring_result"],
              }));
            }
          } else if (evt.type === "error") {
            const detail = evt.data.detail as string;
            const cid = currentIdRef.current;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === cid ? { ...m, content: `Error: ${detail}`, streaming: false } : m
              )
            );
          } else if (evt.type === "done") {
            const cid = currentIdRef.current;
            // Remove trailing placeholder if empty, otherwise finalize it
            setMessages((prev) =>
              prev.flatMap((m) =>
                m.id === cid ? (m.content ? [{ ...m, streaming: false }] : []) : [m]
              )
            );
          }
        }
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : "Connection error";
        const cid = currentIdRef.current;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === cid ? { ...m, content: `Error: ${errMsg}`, streaming: false } : m
          )
        );
      } finally {
        // Finalize or remove trailing placeholder if stream closed without a done event
        setMessages((prev) => {
          const cid = currentIdRef.current;
          return prev.flatMap((m) => {
            if (m.id !== cid) return [m];
            return m.content ? [{ ...m, streaming: false }] : [];
          });
        });
        setIsStreaming(false);
        setActiveNodes([]);
        abortRef.current = null;
      }
    },
    [sessionId, isStreaming]
  );

  const reset = useCallback(() => {
    setMessages([]);
    setActiveNodes([]);
    setCompletedNodes([]);
    setAnalysisData({});
    setIsStreaming(false);
  }, []);

  return {
    messages,
    activeNodes,
    completedNodes,
    isStreaming,
    analysisData,
    send,
    loadHistory,
    reset,
  };
}
