import type { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

const BACKEND = 'http://localhost:8000';

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await params;
  const res = await fetch(`${BACKEND}/api/v1/sessions/${sessionId}/messages`);
  if (!res.ok) return new Response(null, { status: res.status });
  return Response.json(await res.json());
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await params;
  const body = await request.json();

  const backendRes = await fetch(
    `${BACKEND}/api/v1/sessions/${sessionId}/messages`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );

  return new Response(backendRes.body, {
    status: backendRes.status,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
