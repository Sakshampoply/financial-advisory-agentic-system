# Frontend

This document covers the Next.js application structure, custom hooks, SSE streaming and parsing, the rolling-placeholder pattern for multi-agent messages, and the component breakdown.

---

## Table of Contents

1. [Tech Stack](#1-tech-stack)
2. [App Router Structure](#2-app-router-structure)
3. [Custom Hooks](#3-custom-hooks)
4. [Rolling-Placeholder Pattern](#4-rolling-placeholder-pattern)
5. [SSE Parsing](#5-sse-parsing)
6. [API Client](#6-api-client)
7. [ChatLayout](#7-chatlayout)
8. [MarkdownRenderer](#8-markdownrenderer)
9. [SourceCitation Component](#9-sourcecitation-component)
10. [MetricsPanel](#10-metricspanel)
11. [AgentStatusBar](#11-agentstatusbar)

---

## 1. Tech Stack

| Technology | Version | Purpose |
|-----------|:-------:|---------|
| Next.js | 16.2.9 | React framework, App Router |
| React | 19 | UI rendering |
| TypeScript | 5+ | Type safety |
| Tailwind CSS | 3 | Utility-first styling |
| react-markdown | 10 | Markdown rendering |
| remark-gfm | 4 | GitHub Flavored Markdown (tables, strikethrough) |

**Theme**: Dark UI with CSS custom properties defined in `app/globals.css`. Colors: near-black backgrounds (`#0a0a0a`), subtle gray card surfaces, amber accent for source citations.

---

## 2. App Router Structure

```
frontend/
├── app/
│   ├── layout.tsx          ← Global HTML shell, font, global CSS
│   ├── page.tsx            ← Redirects to /chat
│   └── chat/
│       └── page.tsx        ← Renders <ChatLayout>
├── components/
│   ├── chat/
│   │   ├── ChatLayout.tsx         ← Session management orchestration
│   │   ├── ChatWindow.tsx         ← Message list + input
│   │   ├── MessageBubble.tsx      ← Individual message rendering
│   │   ├── MarkdownRenderer.tsx   ← Markdown with source citation injection
│   │   ├── SourceCitation.tsx     ← Amber badge for (Source: ...) citations
│   │   └── AgentStatusBar.tsx     ← Progress indicator during agent runs
│   ├── sidebar/
│   │   └── Sidebar.tsx            ← Session list + document upload
│   └── metrics/
│       ├── MetricsPanel.tsx       ← Risk/allocation/score sidebar
│       ├── ScoreCard.tsx          ← Circular 0-100 score indicator
│       ├── RiskMetrics.tsx        ← Sharpe/vol/drawdown/flags display
│       └── AllocationChart.tsx    ← Horizontal bar chart of holdings
├── hooks/
│   ├── useSession.ts       ← Session list, active session, create/switch
│   ├── useChat.ts          ← Message state, SSE streaming, history loading
│   └── useDocuments.ts     ← Document list, upload
└── lib/
    ├── api.ts              ← Fetch wrappers for all backend endpoints
    ├── sse.ts              ← SSE stream parser (CRLF normalization)
    └── types.ts            ← TypeScript type definitions
```

---

## 3. Custom Hooks

### `useSession` — `hooks/useSession.ts`

Manages the session list and which session is currently active.

**State**: `sessions: Session[]`, `activeSession: Session | null`, `loading: boolean`

**`newSession()`**: Calls `createSession()` API, prepends result to `sessions` list, sets as active. Persists active session ID in `localStorage` so the user's last session is restored on page reload.

**`switchSession(id)`**: Sets the active session without API calls — the session already exists.

**Persistence**: On mount, reads `localStorage.getItem("activeSessionId")` and sets active session if it exists in the loaded list.

---

### `useChat(sessionId)` — `hooks/useChat.ts`

Manages the message list, streaming state, agent progress, and analysis data for a session.

**State**:
- `messages: Message[]` — conversation history including in-progress streaming placeholder
- `activeNodes: string[]` — nodes currently running (show spinner in AgentStatusBar)
- `completedNodes: string[]` — nodes that finished (show checkmark)
- `isStreaming: boolean` — true while SSE connection is open
- `analysisData: AnalysisData | null` — risk metrics, allocation, scoring result

**`send(content, overrideSessionId?)`**: The `overrideSessionId` parameter bypasses stale closure issues. When `handleSend` in `ChatLayout` creates a new session then immediately calls `send`, the `sessionId` prop in `useChat` may not have updated yet (React state batching). Passing `overrideSessionId` directly avoids this.

**`loadHistory(sid)`**: Fetches message history from `GET /sessions/{id}/messages` and populates the messages state. Called when switching sessions.

---

### `useDocuments(sessionId)` — `hooks/useDocuments.ts`

Manages uploaded documents for a session.

**State**: `documents: Document[]`, `uploading: boolean`

**`upload(file, sid)`**: Takes an explicit `sid` parameter (same stale-closure reason as `useChat.send`). Calls `uploadDocument(sid, file)` from `api.ts`. The `sid` parameter is used when `handleFile` in `ChatLayout` creates a new session then immediately uploads — the hook's `sessionId` prop may be stale.

---

## 4. Rolling-Placeholder Pattern

The system produces multiple AI messages per user turn (e.g., `intake` asks a follow-up question, then `advisor_copilot` delivers the full analysis). The frontend needs to display these as separate message bubbles and stream tokens into the correct one.

**The problem**: SSE `token` events arrive while `advisor_copilot`'s LLM is still streaming, but `message` events (with final content) arrive only after the graph completes. There's no upfront count of how many messages to expect.

**The solution** — a rolling placeholder:

1. When the first `token` event arrives, create a new `Message` object with a generated UUID and empty `content`. Add it to the messages array. This is the "active placeholder".

2. Each subsequent `token` event appends `chunk.content` to the active placeholder's content → live typing effect.

3. When a `message` event arrives (graph completed, final content known):
   - **Finalize** the current placeholder: replace its content with the `message` event's authoritative content (which includes the disclaimer from `guardrail_output`)
   - **Create a new empty trailing placeholder** — in case more `message` events arrive (e.g., multi-agent responses)

4. When the `done` event arrives:
   - Remove any empty trailing placeholder (no content was streamed into it)

5. **`finally` block**: If the SSE stream closes without a `done` event, clean up any empty placeholder.

**`currentIdRef`**: A `useRef` tracks the UUID of the current active placeholder. This avoids stale closure issues — the ref is mutable and always reflects the current placeholder ID.

---

## 5. SSE Parsing

**File**: `lib/sse.ts`

```typescript
export async function* streamSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // CRITICAL: sse_starlette v3.4.4 sends \r\n CRLF line endings.
    // Normalize to \n before parsing so split("\n\n") works correctly.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";  // last part may be incomplete

    for (const part of parts) {
      if (!part.trim()) continue;
      const event: Partial<SSEEvent> = {};
      for (const line of part.split("\n")) {
        if (line.startsWith("event: ")) event.event = line.slice(7);
        if (line.startsWith("data: "))  event.data  = line.slice(6);
      }
      if (event.event && event.data) {
        yield event as SSEEvent;
      }
    }
  }
}
```

**Why CRLF normalization is critical**: `sse_starlette` (the library used on the backend) defaults to `\r\n` CRLF line endings per the SSE specification. The SSE spec says events are separated by `\n\n` or `\r\n\r\n`. The frontend split on `"\n\n"` — which never matches `"\r\n\r\n"` because the two `\n` chars are separated by `\r`. Without normalization, zero events would be parsed and the entire UI would silently receive nothing. The backend also sets `sep="\n"` on `EventSourceResponse` for redundancy.

**`{stream: true}`** in `decoder.decode()`: Tells the TextDecoder not to flush its internal buffer — important for multi-byte UTF-8 characters that may be split across chunk boundaries.

---

## 6. API Client

**File**: `lib/api.ts`

```typescript
const BASE = "/api/v1";                           // through Next.js proxy
const STREAM_BASE = "http://localhost:8000/api/v1"; // direct to backend
```

Non-streaming calls (`createSession`, `listSessions`, `getSessionMessages`, `uploadDocument`, `listDocuments`) use `BASE` — they go through the Next.js `rewrites` proxy configured in `next.config.ts`.

`sendMessage` uses `STREAM_BASE` to avoid proxy buffering. See [api.md → Why Direct SSE for Streaming](api.md#6-why-direct-sse-for-streaming).

**`listDocuments` field mapping**: The backend returns `doc_id` and `uploaded_at`. The frontend `Document` type uses `id` and `created_at`. The mapping happens in `api.ts`:

```typescript
return data.map((d) => ({
  id: d.doc_id,
  filename: d.filename,
  session_id: sessionId,
  created_at: d.uploaded_at,
}));
```

Without this mapping, `document.id` would be `undefined`, causing React `key={undefined}` warnings and broken list rendering.

---

## 7. ChatLayout

**File**: `components/chat/ChatLayout.tsx`

Orchestrates sessions, messages, documents, and the main chat UI.

**`initialLoadRef`**: Prevents loading history more than once when the session is first set. The pattern:
```typescript
const initialLoadRef = useRef(false);
useEffect(() => {
  if (!loading && activeSession && !initialLoadRef.current) {
    initialLoadRef.current = true;
    loadHistory(activeSession.id);
    loadDocuments(activeSession.id);
  }
}, [loading, activeSession]);
```
Without this guard, the effect would fire on every render while `loading` is false and `activeSession` is set.

**`handleSend`**: Creates a new session if none exists, then calls `chat.send(content, newSession.id)` passing the session ID directly to avoid stale closure.

**`handleFile`**: Same pattern — creates session if needed, then calls `documents.upload(file, newSession.id)`.

**`handleSelectSession`**: Explicitly calls `loadHistory(id)` and `loadDocuments(id)` when switching to a different session.

---

## 8. MarkdownRenderer

**File**: `components/chat/MarkdownRenderer.tsx`

Renders advisor responses as formatted markdown with source citations converted to amber badges.

### The problem with react-markdown v10

The native approach to transforming text inside markdown components is to traverse the component tree via `processChildren`. In react-markdown v10, text nodes adjacent to inline elements arrive as React elements (not strings), so regex matching on `.children` is unreliable — the string may be split across multiple sibling elements.

### The solution: string preprocessing

Before the content reaches ReactMarkdown, transform all `(Source: filename.txt)` patterns into inline code markers:

```typescript
function injectSourceMarkers(content: string): string {
  return content.replace(
    /\(Source:\s*([^)\n]+?)\s*\)/g,
    (_, filename) => `\`[[src:${filename.trim()}]]\``
  );
}
```

This converts `(Source: SEC_ETF_Guide.txt)` → `` `[[src:SEC_ETF_Guide.txt]]` ``

ReactMarkdown then parses the backtick-wrapped text as an inline `<code>` node. A custom `code` component detects the `[[src:...]]` pattern:

```typescript
code({ children, className }: any) {
  const text = typeof children === "string" ? children : String(children ?? "");
  if (!className && text.startsWith("[[src:") && text.endsWith("]]")) {
    return <SourceCitation filename={text.slice(6, -2)} />;
  }
  return <code className={className}>{children}</code>;
}
```

The `!className` check distinguishes inline code (`<code>` without a class) from fenced code blocks (which have `language-xxx` className from remark-gfm). Source citation markers are always inline.

---

## 9. SourceCitation Component

**File**: `components/chat/SourceCitation.tsx`

Renders a single `(Source: filename.txt)` citation as an inline amber badge:

```
[ 📄 SEC_ETF_Guide.txt ]
```

Implemented as an inline `<span>` (not `<div>`) so it flows within paragraph text without breaking line layout. Styled with amber background, dark brown text, rounded pill shape, and a small document icon SVG.

---

## 10. MetricsPanel

**File**: `components/metrics/MetricsPanel.tsx`

The right sidebar that displays quantitative analysis results. Only renders when `hasData` is true:

```typescript
const hasData = risk_metrics || scoring_result || allocation_result;
if (!hasData) return null;
```

Data arrives from the SSE `state` event after the graph completes. The `useChat` hook stores it in `analysisData` and passes it to `ChatLayout`, which passes it to `MetricsPanel`.

**Slide-in animation**: Uses a CSS transition (`translate-x-0` from `translate-x-full`) triggered when `hasData` becomes true — the panel slides in from the right.

**Sub-components**:

### ScoreCard

Circular indicator showing composite portfolio score (0–100). Uses an SVG circle with `strokeDasharray` and `strokeDashoffset` to draw the arc proportional to the score. Color gradient: red (0–40) → amber (40–70) → green (70–100).

### RiskMetrics

Displays:
- Sharpe ratio with sign indicator (green if > 1.0, amber if > 0, red if negative)
- Annualized volatility as percentage
- Max drawdown as percentage (always negative — red)
- Risk flags as a bulleted list

### AllocationChart

Horizontal bar chart of the top-8 holdings by weight from `allocation_result.weights`. Each bar is a `<div>` with `width` set to `${weight * 100}%`. Sorted descending by weight. Truncated at 8 items to avoid overflow.

---

## 11. AgentStatusBar

**File**: `components/chat/AgentStatusBar.tsx`

Displays which agents are running or have completed during a graph execution.

**Active nodes**: Show with an animated spinner (CSS `animate-spin` on an SVG circle)

**Completed nodes**: Show with a static checkmark

**Human-readable labels** (`AGENT_LABELS` map):
| Node key | Display label |
|----------|--------------|
| `guardrail_input` | Checking input |
| `intake` | Collecting profile |
| `document_intelligence` | Analyzing document |
| `profile_builder` | Building profile |
| `risk_assessment` | Assessing risk |
| `strategy` | Optimizing allocation |
| `scoring` | Scoring portfolio |
| `advisor_copilot` | Generating advice |

The bar is visible during streaming and disappears once `isStreaming` becomes false and `done` event has been received.
