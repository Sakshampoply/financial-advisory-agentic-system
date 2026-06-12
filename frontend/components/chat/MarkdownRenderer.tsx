"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SourceCitation } from "./SourceCitation";

// Before markdown parsing, replace (Source: filename) patterns with inline code markers.
// Using \`[[src:filename]]\` so the markdown parser turns them into <code> nodes,
// which our custom code renderer then converts to SourceCitation badges.
function injectSourceMarkers(content: string): string {
  return content.replace(
    /\(Source:\s*([^)\n]+?)\s*\)/g,
    (_, filename) => `\`[[src:${filename.trim()}]]\``
  );
}

export function MarkdownRenderer({ content, streaming }: { content: string; streaming?: boolean }) {
  return (
    <div className={`prose${streaming ? " typing-cursor" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          code({ children, className }: any) {
            const text = typeof children === "string" ? children : String(children ?? "");
            // Render source citation markers as styled badges
            if (!className && text.startsWith("[[src:") && text.endsWith("]]")) {
              return <SourceCitation filename={text.slice(6, -2)} />;
            }
            return <code className={className}>{children}</code>;
          },
        }}
      >
        {injectSourceMarkers(content)}
      </ReactMarkdown>
    </div>
  );
}
