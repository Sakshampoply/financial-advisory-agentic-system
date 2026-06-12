"use client";
import { useState, useEffect, useCallback } from "react";
import { createSession, listSessions } from "@/lib/api";
import type { Session } from "@/lib/types";

const STORAGE_KEY = "fa_session_id";

export function useSession() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  const loadSessions = useCallback(async () => {
    const all = await listSessions();
    setSessions(all);
    return all;
  }, []);

  useEffect(() => {
    (async () => {
      const all = await loadSessions();
      const storedId = localStorage.getItem(STORAGE_KEY);
      const found = all.find((s) => s.id === storedId);
      if (found) {
        setActiveSession(found);
      }
      setLoading(false);
    })();
  }, [loadSessions]);

  const newSession = useCallback(async () => {
    const session = await createSession();
    localStorage.setItem(STORAGE_KEY, session.id);
    setSessions((prev) => {
      const exists = prev.some((s) => s.id === session.id);
      return exists ? prev : [session, ...prev];
    });
    setActiveSession(session);
    return session;
  }, []);

  const switchSession = useCallback((session: Session) => {
    localStorage.setItem(STORAGE_KEY, session.id);
    setActiveSession(session);
  }, []);

  return { sessions, activeSession, loading, newSession, switchSession, loadSessions };
}
