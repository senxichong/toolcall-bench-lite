export type Metric = { passed: number; total: number; rate: number }

export type Summary = {
  total_tasks: number
  tool_selection_accuracy: Metric
  argument_match: Metric
  schema_validity: Metric
  execution_success: Metric
  final_answer_accuracy: Metric
  overall_success: Metric
}

export type DatasetTask = {
  task_id: string
  prompt: string
  available_tools: string[]
  expected_tool: string | null
  expected_arguments: Record<string, unknown>
  expected_answer: unknown
  answer_match?: 'auto' | 'exact' | 'json' | 'numeric' | 'contains'
  metadata?: Record<string, unknown>
}

export type Dataset = {
  id: string
  name: string
  version: string
  description?: string | null
  task_count: number
  created_at: string
  tasks?: DatasetTask[]
}

export type BackendConfig = {
  type: 'mock' | 'openai' | 'local'
  model?: string
}

export type Job = {
  id: string
  status: string
  dataset_id: string
  backend: BackendConfig
  run_id: string | null
  error: string | null
  created_at: string
  updated_at: string
}

export type Run = {
  id: string
  job_id: string
  dataset_id: string
  dataset_name: string
  dataset_version: string
  model: string
  created_at: string
  summary: Summary
}

export type AgentTurn = {
  content?: string | null
  tool_call?: { name: string; arguments: Record<string, unknown>; id?: string | null } | null
  raw?: unknown
  schema_valid?: boolean
  error?: string | null
}

export type TaskResult = {
  task_id?: string
  task?: DatasetTask
  turn?: AgentTurn
  execution?: { success: boolean; value?: unknown; error?: string | null; tool_name?: string | null; arguments?: Record<string, unknown> | null } | null
  final_turn?: AgentTurn | null
  evaluation: {
    task_id: string
    tool_selection_correct: boolean
    arguments_correct: boolean
    schema_valid: boolean
    execution_success: boolean
    final_answer_correct: boolean
    overall_success?: boolean
    details?: Record<string, unknown>
    errors?: string[]
  }
  latency_ms?: number | null
  trace?: unknown[]
}

export type RunDetail = Run & {
  report: {
    model: string
    created_at: string
    benchmark?: string | null
    metadata?: Record<string, unknown>
    summary?: Summary
    results: TaskResult[]
  }
}

export type Badcase = {
  task_id: string
  primary_category: string
  categories: string[]
  reason: string
  trace: unknown[]
  [key: string]: unknown
}

export type ModelMetrics = {
  model: string
  runs: number
  total_tasks: number
  summary: Summary
}

export type Stats = {
  total_runs: number
  total_jobs: number
  failed_jobs: number
  comparison_dataset: Dataset | null
  model_metrics: ModelMetrics[]
}
