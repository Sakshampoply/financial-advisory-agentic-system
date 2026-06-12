"use client";
import { useState, useRef, KeyboardEvent } from "react";
import { FileUploadButton } from "./FileUploadButton";
import { Spinner } from "@/components/ui/Spinner";

interface InputBarProps {
  onSend: (content: string) => void;
  onFile: (file: File) => void;
  isStreaming: boolean;
  uploading: boolean;
  disabled: boolean;
}

export function InputBar({ onSend, onFile, isStreaming, uploading, disabled }: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSend() {
    const trimmed = value.trim();
    if (!trimmed || isStreaming || disabled) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }

  const canSend = value.trim().length > 0 && !isStreaming && !disabled;

  return (
    <div
      className="flex-shrink-0 px-4 py-3 border-t"
      style={{ borderColor: "#1E2A3A" }}
    >
      <div
        className="flex items-end gap-2 rounded-xl px-3 py-2"
        style={{ background: "#0F1520", border: "1px solid #1E2A3A" }}
      >
        <FileUploadButton onFile={onFile} uploading={uploading} />
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? "Select or create a session to start..." : "Type a message... (Shift+Enter for new line)"}
          disabled={disabled || isStreaming}
          rows={1}
          className="flex-1 bg-transparent resize-none outline-none text-sm py-0.5"
          style={{
            color: "#F0F4F8",
            caretColor: "#E8A020",
            maxHeight: "180px",
          }}
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={!canSend}
          title="Send message"
          className="flex items-center justify-center w-8 h-8 rounded-lg transition-all cursor-pointer disabled:opacity-30"
          style={{
            background: canSend ? "#E8A020" : "#1E2A3A",
            color: canSend ? "#080C14" : "#6B7E96",
          }}
        >
          {isStreaming ? (
            <Spinner size={14} />
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>
      <p className="text-center text-xs mt-2" style={{ color: "#6B7E96" }}>
        AI responses may contain errors. Not professional financial advice.
      </p>
    </div>
  );
}
