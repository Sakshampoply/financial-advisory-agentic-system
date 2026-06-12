"use client";
import { ScoreCard } from "./ScoreCard";
import { RiskMetrics } from "./RiskMetrics";
import { AllocationChart } from "./AllocationChart";
import type { AnalysisData } from "@/lib/types";

interface MetricsPanelProps {
  data: AnalysisData;
}

export function MetricsPanel({ data }: MetricsPanelProps) {
  const hasData =
    data.risk_metrics || data.scoring_result || data.allocation_result;

  if (!hasData) return null;

  return (
    <aside
      className="metrics-slide-in w-[280px] flex-shrink-0 border-l overflow-y-auto"
      style={{ borderColor: "#1E2A3A", background: "#0F1520" }}
    >
      <div className="p-4 space-y-5">
        <div
          className="pb-3 border-b"
          style={{ borderColor: "#1E2A3A" }}
        >
          <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: "#6B7E96" }}>
            Analysis
          </p>
        </div>

        {data.scoring_result && (
          <div className="pb-4 border-b" style={{ borderColor: "#1E2A3A" }}>
            <ScoreCard score={data.scoring_result.composite_score} />
            {Object.keys(data.scoring_result.breakdown).length > 0 && (
              <div className="mt-3 space-y-1.5">
                {Object.entries(data.scoring_result.breakdown).map(([key, val]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span style={{ color: "#6B7E96" }}>
                      {key.replace("_score", "").replace(/_/g, " ")}
                    </span>
                    <span style={{ color: "#F0F4F8" }}>{Math.round(val)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {data.risk_metrics && (
          <div className="pb-4 border-b" style={{ borderColor: "#1E2A3A" }}>
            <RiskMetrics metrics={data.risk_metrics} />
          </div>
        )}

        {data.allocation_result && (
          <div>
            <AllocationChart weights={data.allocation_result.weights} />
            {data.allocation_result.expected_return && (
              <div className="mt-3 pt-3 border-t space-y-1" style={{ borderColor: "#1E2A3A" }}>
                <div className="flex justify-between text-xs">
                  <span style={{ color: "#6B7E96" }}>Expected Return</span>
                  <span style={{ color: "#22C55E" }}>
                    {(data.allocation_result.expected_return * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span style={{ color: "#6B7E96" }}>Expected Vol.</span>
                  <span style={{ color: "#F0F4F8" }}>
                    {(data.allocation_result.expected_volatility * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
