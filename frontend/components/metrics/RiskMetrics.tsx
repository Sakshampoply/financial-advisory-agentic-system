"use client";
import type { RiskMetrics as RiskMetricsType } from "@/lib/types";

interface StatProps {
  label: string;
  value: string;
  positive?: boolean;
  negative?: boolean;
}

function Stat({ label, value, positive, negative }: StatProps) {
  const color = positive ? "#22C55E" : negative ? "#EF4444" : "#F0F4F8";
  return (
    <div className="flex justify-between items-center py-1.5">
      <span className="text-xs" style={{ color: "#6B7E96" }}>
        {label}
      </span>
      <span className="text-sm font-semibold tabular-nums" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

export function RiskMetrics({ metrics }: { metrics: RiskMetricsType }) {
  const { sharpe_ratio, volatility, max_drawdown, risk_flags } = metrics;

  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "#6B7E96" }}>
        Risk Metrics
      </p>
      <div className="divide-y" style={{ borderColor: "#1E2A3A" }}>
        <Stat
          label="Sharpe Ratio"
          value={sharpe_ratio.toFixed(2)}
          positive={sharpe_ratio >= 1}
          negative={sharpe_ratio < 0.5}
        />
        <Stat
          label="Volatility"
          value={`${(volatility * 100).toFixed(1)}%`}
          negative={volatility > 0.25}
        />
        <Stat
          label="Max Drawdown"
          value={`${(max_drawdown * 100).toFixed(1)}%`}
          negative={max_drawdown < -0.2}
        />
      </div>
      {risk_flags.length > 0 && (
        <div className="mt-2 pt-2 border-t" style={{ borderColor: "#1E2A3A" }}>
          <p className="text-xs font-semibold uppercase tracking-wide mb-1.5" style={{ color: "#EF4444" }}>
            Risk Flags
          </p>
          <ul className="space-y-1">
            {risk_flags.map((flag, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs" style={{ color: "#6B7E96" }}>
                <span className="mt-0.5" style={{ color: "#EF4444" }}>•</span>
                {flag}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
