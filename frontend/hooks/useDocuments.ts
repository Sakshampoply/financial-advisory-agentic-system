"use client";
import { useState, useCallback } from "react";
import { uploadDocument, listDocuments } from "@/lib/api";
import type { Document } from "@/lib/types";

export function useDocuments(sessionId: string | null) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const loadDocuments = useCallback(async (sid: string) => {
    const docs = await listDocuments(sid);
    setDocuments(docs);
  }, []);

  const upload = useCallback(
    async (file: File, overrideSessionId?: string) => {
      const sid = overrideSessionId ?? sessionId;
      if (!sid) return;
      setUploading(true);
      setUploadError(null);
      try {
        const doc = await uploadDocument(sid, file);
        setDocuments((prev) => [...prev, doc]);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [sessionId]
  );

  return { documents, uploading, uploadError, upload, loadDocuments };
}
