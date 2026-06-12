"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html>
      <body
        style={{
          background: "#080C14",
          color: "#F0F4F8",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          fontFamily: "system-ui, sans-serif",
          flexDirection: "column",
          gap: "16px",
        }}
      >
        <p style={{ color: "#EF4444" }}>Something went wrong.</p>
        <button
          onClick={reset}
          style={{
            background: "#E8A020",
            color: "#080C14",
            border: "none",
            padding: "8px 16px",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Try again
        </button>
      </body>
    </html>
  );
}
