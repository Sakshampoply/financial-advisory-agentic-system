import type { SSEEvent, SSEEventType } from "./types";

export function parseSSEChunk(chunk: string): SSEEvent[] {
  const events: SSEEvent[] = [];
  const blocks = chunk.split("\n\n");
  for (const block of blocks) {
    if (!block.trim()) continue;
    const lines = block.split("\n");
    let eventType: SSEEventType = "message";
    let dataStr = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim() as SSEEventType;
      } else if (line.startsWith("data: ")) {
        dataStr = line.slice(6).trim();
      }
    }
    if (!dataStr) continue;
    try {
      const data = JSON.parse(dataStr);
      events.push({ type: eventType, data });
    } catch {
      // skip malformed
    }
  }
  return events;
}

export async function* streamSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const events = parseSSEChunk(part + "\n\n");
        for (const evt of events) yield evt;
      }
    }
    if (buffer.trim()) {
      const events = parseSSEChunk(buffer);
      for (const evt of events) yield evt;
    }
  } finally {
    reader.releaseLock();
  }
}
