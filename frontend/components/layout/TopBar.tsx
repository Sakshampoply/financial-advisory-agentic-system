"use client";
import { Button } from "@/components/ui/Button";
import { Plus } from "lucide-react";

interface TopBarProps {
  onNewChat: () => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export function TopBar({ onNewChat, sidebarOpen, onToggleSidebar }: TopBarProps) {
  return (
    <header
      className="flex-shrink-0 h-[52px] flex items-center justify-between px-4 border-b"
      style={{ borderColor: "#1E2A3A", background: "#080C14" }}
    >
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="p-1.5 rounded transition-colors cursor-pointer md:hidden"
          style={{ color: "#6B7E96" }}
          title="Toggle sidebar"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: "#E8A020" }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#080C14" strokeWidth="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <span className="font-semibold text-sm tracking-tight" style={{ color: "#F0F4F8" }}>
            Financial Advisor <span style={{ color: "#E8A020" }}>AI</span>
          </span>
        </div>
      </div>
      <Button variant="primary" size="sm" onClick={onNewChat}>
        <Plus size={14} />
        New chat
      </Button>
    </header>
  );
}
