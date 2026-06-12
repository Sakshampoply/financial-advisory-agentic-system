"use client";

const COLORS = ["#E8A020", "#22C55E", "#6B7E96", "#EF4444", "#8B5CF6", "#3B82F6", "#EC4899"];

interface AllocationChartProps {
  weights: Record<string, number>;
}

export function AllocationChart({ weights }: AllocationChartProps) {
  const sorted = Object.entries(weights)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8);

  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "#6B7E96" }}>
        Allocation
      </p>
      <div className="space-y-2">
        {sorted.map(([ticker, weight], i) => (
          <div key={ticker} className="space-y-0.5">
            <div className="flex justify-between text-xs">
              <span style={{ color: "#F0F4F8" }}>{ticker}</span>
              <span style={{ color: "#6B7E96" }}>{(weight * 100).toFixed(1)}%</span>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#1E2A3A" }}>
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${weight * 100}%`,
                  background: COLORS[i % COLORS.length],
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
