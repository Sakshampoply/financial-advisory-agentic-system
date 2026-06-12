import type { Session, Document } from "./types";

const BASE = "/api/v1";
const STREAM_BASE = "http://localhost:8000/api/v1";

export async function createSession(): Promise<Session> {
  const res = await fetch(`${BASE}/sessions`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to create session: ${res.status}`);
  return res.json();
}

export async function listSessions(): Promise<Session[]> {
  const res = await fetch(`${BASE}/sessions`);
  if (!res.ok) return [];
  return res.json();
}

export async function getSessionMessages(sessionId: string): Promise<{ role: string; content: string; agent?: string }[]> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/messages`);
  if (!res.ok) return [];
  return res.json();
}

export async function uploadDocument(sessionId: string, file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/sessions/${sessionId}/documents`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function listDocuments(sessionId: string): Promise<Document[]> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/documents`);
  if (!res.ok) return [];
  const data: { doc_id: string; filename: string; uploaded_at: string }[] = await res.json();
  return data.map((d) => ({
    id: d.doc_id,
    filename: d.filename,
    session_id: sessionId,
    created_at: d.uploaded_at,
  }));
}

export function sendMessage(sessionId: string, content: string): ReadableStream<Uint8Array> {
  const controller = new AbortController();
  const stream = new ReadableStream<Uint8Array>({
    async start(ctrl) {
      try {
        const res = await fetch(`${STREAM_BASE}/sessions/${sessionId}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          const errText = await res.text().catch(() => "Unknown error");
          ctrl.error(new Error(errText));
          return;
        }
        const reader = res.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          ctrl.enqueue(value);
        }
        ctrl.close();
      } catch (err) {
        ctrl.error(err);
      }
    },
    cancel() {
      controller.abort();
    },
  });
  return stream;
}
