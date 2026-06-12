"use client";

interface ScoreCardProps {
  score: number;
}

export function ScoreCard({ score }: ScoreCardProps) {
  const clamped = Math.max(0, Math.min(100, score));
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const strokeDash = (clamped / 100) * circumference;

  const color =
    clamped >= 75 ? "#22C55E" : clamped >= 50 ? "#E8A020" : "#EF4444";

  const label =
    clamped >= 80 ? "Excellent" : clamped >= 65 ? "Good" : clamped >= 50 ? "Fair" : "Needs Work";

  return (
    <div className="flex flex-col items-center gap-2 py-2">
      <div className="relative">
        <svg width="96" height="96" viewBox="0 0 96 96">
          <circle
            cx="48"
            cy="48"
            r={radius}
            fill="none"
            stroke="#1E2A3A"
            strokeWidth="8"
          />
          <circle
            cx="48"
            cy="48"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${strokeDash} ${circumference}`}
            strokeDashoffset="0"
            transform="rotate(-90 48 48)"
            style={{ transition: "stroke-dasharray 0.6s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold" style={{ color }}>
            {Math.round(clamped)}
          </span>
          <span className="text-xs" style={{ color: "#6B7E96" }}>
            / 100
          </span>
        </div>
      </div>
      <div>
        <p className="text-sm font-semibold text-center" style={{ color }}>
          {label}
        </p>
        <p className="text-xs text-center" style={{ color: "#6B7E96" }}>
          Portfolio Score
        </p>
      </div>
    </div>
  );
}
