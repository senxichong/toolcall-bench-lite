import type { BackendConfig, Badcase, Dataset, DatasetTask, Job, Run, RunDetail, Stats } from './types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')
const API_KEY_STORAGE = 'toolcall-bench-api-key'

export function getApiKey(): string {
  try { return sessionStorage.getItem(API_KEY_STORAGE) || '' } catch { return '' }
}

export function setApiKey(key: string): void {
  try {
    if (key.trim()) sessionStorage.setItem(API_KEY_STORAGE, key.trim())
    else sessionStorage.removeItem(API_KEY_STORAGE)
  } catch {
    // In private contexts with disabled storage, the UI can still run without a key.
  }
}

function headers(body: boolean): HeadersInit {
  const key = getApiKey()
  return { Accept: 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}), ...(key ? { 'X-API-Key': key } : {}) }
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { ...headers(Boolean(init?.body)), ...init?.headers },
    })
  } catch {
    throw new ApiError('Cannot connect to the API. Check that the backend is running.', 0)
  }
  if (!response.ok) {
    let detail: unknown
    try {
      const body = await response.json()
      detail = body.detail ?? body.error ?? body.message
    } catch {
      // The service may return an empty or non-JSON error response.
    }
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item) => item.msg || JSON.stringify(item)).join('; ') : `Request failed (${response.status})`
    throw new ApiError(message, response.status)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  stats: (datasetId?: string) => request<Stats>(datasetId ? `/stats?dataset_id=${encodeURIComponent(datasetId)}` : '/stats'),
  runs: async () => (await request<{ items: Run[] }>('/runs')).items,
  run: (id: string) => request<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  badcases: async (id: string) => (await request<{ items: Badcase[] }>(`/runs/${encodeURIComponent(id)}/badcases`)).items,
  datasets: async () => (await request<{ items: Dataset[] }>('/datasets')).items,
  dataset: (id: string) => request<Dataset>(`/datasets/${encodeURIComponent(id)}`),
  createDataset: (body: { name: string; version: string; description: string; tasks: DatasetTask[] }) => request<Dataset>('/datasets', { method: 'POST', body: JSON.stringify(body) }),
  createEvaluation: (dataset_id: string, backend: BackendConfig) => request<Job>('/evaluations', { method: 'POST', body: JSON.stringify({ dataset_id, backend }) }),
  job: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
  exportDataset: async (id: string, name: string, version: string): Promise<void> => {
    let response: Response
    try { response = await fetch(`${API_BASE}/datasets/${encodeURIComponent(id)}/export`, { headers: headers(false) }) }
    catch { throw new ApiError('Cannot connect to the API to export the dataset.', 0) }
    if (!response.ok) throw new ApiError(`Dataset export failed (${response.status})`, response.status)
    const url = URL.createObjectURL(await response.blob())
    const link = document.createElement('a')
    link.href = url
    link.download = `${name}-${version}.jsonl`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  },
}
