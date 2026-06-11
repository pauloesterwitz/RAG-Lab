// Thin API client for the FastAPI backend.
const BASE = ''

async function j(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail || detail } catch (e) {}
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  approaches: () => j('GET', '/api/approaches'),
  status: () => j('GET', '/api/status'),
  job: () => j('GET', '/api/job'),
  reembed: (rebuild_graph) => j('POST', '/api/reembed', { rebuild_graph }),
  synth: (num) => j('POST', '/api/synth', { num }),
  evalRun: (approaches) => j('POST', '/api/eval', { approaches }),
  full: (num) => j('POST', '/api/full', { num }),
  chat: (approach, query) => j('POST', '/api/chat', { approach, query }),
  metrics: () => j('GET', '/api/metrics'),
  goldens: () => j('GET', '/api/goldens'),
}
