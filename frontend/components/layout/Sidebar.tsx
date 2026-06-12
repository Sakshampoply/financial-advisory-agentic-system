"use client";
import type { Session, Document } from "@/lib/types";

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

interface SidebarProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSelectSession: (session: Session) => void;
  documents: Document[];
  open: boolean;
}

export function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  documents,
  open,
}: SidebarProps) {
  return (
    <aside
      className={`flex-shrink-0 w-[240px] border-r flex flex-col overflow-hidden transition-all duration-200 ${
        open ? "flex" : "hidden md:flex"
      }`}
      style={{ borderColor: "#1E2A3A", background: "#0F1520" }}
    >
      <div className="flex-1 overflow-y-auto py-3">
        {sessions.length === 0 ? (
          <div className="px-4 py-6 text-xs text-center" style={{ color: "#6B7E96" }}>
            No conversations yet.
            <br />
            Start a new chat to begin.
          </div>
        ) : (
          <div>
            <p
              className="px-4 py-1.5 text-xs font-semibold uppercase tracking-wide"
              style={{ color: "#6B7E96" }}
            >
              Conversations
            </p>
            {sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <button
                  key={session.id}
                  type="button"
                  onClick={() => onSelectSession(session)}
                  className="w-full text-left px-4 py-2.5 transition-colors cursor-pointer"
                  style={{
                    background: isActive ? "#1E2A3A" : "transparent",
                    borderLeft: isActive ? "2px solid #E8A020" : "2px solid transparent",
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className="text-xs font-medium truncate"
                      style={{ color: isActive ? "#F0F4F8" : "#6B7E96" }}
                    >
                      Session {session.id.slice(0, 8)}…
                    </span>
                    <span className="text-xs flex-shrink-0" style={{ color: "#6B7E96" }}>
                      {relativeTime(session.created_at)}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {documents.length > 0 && (
          <div className="mt-4 border-t pt-3" style={{ borderColor: "#1E2A3A" }}>
            <p
              className="px-4 py-1.5 text-xs font-semibold uppercase tracking-wide"
              style={{ color: "#6B7E96" }}
            >
              Uploaded Documents
            </p>
            <div className="px-4 space-y-1">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center gap-2 py-1.5"
                >
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#E8A020"
                    strokeWidth="2"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                  <span className="text-xs truncate" style={{ color: "#6B7E96" }}>
                    {doc.filename}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
