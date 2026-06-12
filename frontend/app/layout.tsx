import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial Advisor AI",
  description: "AI-powered financial advisory system",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <body className="h-full" suppressHydrationWarning>{children}</body>
    </html>
  );
}
