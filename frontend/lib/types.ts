export interface Session {
  id: string;
  created_at: string;
  updated_at: string;
  langgraph_thread_id: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  agent?: string;
  streaming?: boolean;
  timestamp: number;
}

export interface Document {
  id: string;
  filename: string;
  session_id: string;
  created_at: string;
}

export interface RiskMetrics {
  sharpe_ratio: number;
  volatility: number;
  max_drawdown: number;
  risk_flags: string[];
}

export interface AllocationResult {
  weights: Record<string, number>;
  expected_return: number;
  expected_volatility: number;
  strategy_rationale: string;
}

export interface ScoringResult {
  composite_score: number;
  breakdown: Record<string, number>;
}

export interface AnalysisData {
  risk_metrics?: RiskMetrics;
  allocation_result?: AllocationResult;
  scoring_result?: ScoringResult;
}

export type SSEEventType = "node_start" | "node_complete" | "token" | "message" | "state" | "done" | "error";

export interface SSEEvent {
  type: SSEEventType;
  data: Record<string, unknown>;
}
