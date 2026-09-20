import { useCallback, useEffect, useMemo, useState, type ChangeEvent, type FormEvent, type ReactNode } from 'react'
import { api, getApiKey, setApiKey } from './api'
import type { BackendConfig, Badcase, Dataset, DatasetTask, Job, Metric, ModelMetrics, Run, RunDetail, Stats, Summary, TaskResult } from './types'

type View = 'overview' | 'comparison' | 'runs' | 'failures' | 'datasets'

const METRICS: { key: keyof Omit<Summary, 'total_tasks'>; label: string; short: string }[] = [
  { key: 'tool_selection_accuracy', label: 'Tool selection accuracy', short: 'Tool selection' },
  { key: 'argument_match', label: 'Argument correctness', short: 'Arguments' },
  { key: 'schema_validity', label: 'Schema validity', short: 'Schema' },
  { key: 'execution_success', label: 'Execution success', short: 'Execution' },
  { key: 'final_answer_accuracy', label: 'Final answer quality', short: 'Final answer' },
  { key: 'overall_success', label: 'Overall success', short: 'Overall' },
]

const NAV: { id: View; label: string; icon: string }[] = [
  { id: 'overview', label: 'Overview', icon: '◫' },
  { id: 'comparison', label: 'Model comparison', icon: '▥' },
  { id: 'runs', label: 'Evaluation runs', icon: '▶' },
  { id: 'failures', label: 'Failure analysis', icon: '◇' },
  { id: 'datasets', label: 'Datasets', icon: '▤' },
]

const pct = (rate: number | undefined) => rate == null || !Number.isFinite(rate) ? '—' : `${(rate * 100).toFixed(1)}%`
const count = (value: number | undefined) => value == null ? '—' : new Intl.NumberFormat().format(value)
const formatDate = (value: string | undefined) => {
  if (!value) return '—'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(parsed)
}
const pretty = (value: unknown) => value === undefined ? '—' : typeof value === 'string' ? value : JSON.stringify(value, null, 2)
const taskId = (result: TaskResult) => result.task_id || result.task?.task_id || result.evaluation.task_id
const isSuccess = (result: TaskResult) => result.evaluation.overall_success ?? [result.evaluation.tool_selection_correct, result.evaluation.arguments_correct, result.evaluation.schema_valid, result.evaluation.execution_success, result.evaluation.final_answer_correct].every(Boolean)
const categoryLabel = (value: string) => value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
const traceTokens = (result: TaskResult): number | null => {
  const values = (result.trace || []).map((step) => step && typeof step === 'object' && 'token_usage' in step ? (step as { token_usage?: { total_tokens?: unknown } }).token_usage?.total_tokens : null)
  const known = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  return known.length ? known.reduce((sum, value) => sum + value, 0) : null
}

function EmptyState({ icon = '◫', title, children }: { icon?: string; title: string; children?: ReactNode }) {
  return <div className="empty-state"><span className="empty-icon" aria-hidden="true">{icon}</span><h3>{title}</h3>{children && <p>{children}</p>}</div>
}

function ErrorBanner({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="alert error" role="alert"><span aria-hidden="true">!</span><div>{message}</div>{retry && <button type="button" className="text-button" onClick={retry}>Retry</button>}</div>
}

function MetricBar({ label, metric, compact = false }: { label: string; metric?: Metric; compact?: boolean }) {
  return <div className={`metric-row ${compact ? 'compact' : ''}`}>
    <div className="metric-row-top"><span>{label}</span><strong>{metric ? pct(metric.rate) : '—'}</strong></div>
    <div className="bar-track" role="img" aria-label={`${label}: ${metric ? pct(metric.rate) : 'unavailable'}`}><div className="bar-fill" style={{ width: `${Math.min(100, Math.max(0, (metric?.rate || 0) * 100))}%` }} /></div>
    {!compact && <small>{metric ? `${count(metric.passed)} of ${count(metric.total)} passed` : 'No data available'}</small>}
  </div>
}

function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase()
  const tone = ['succeeded', 'completed', 'success', 'finished'].includes(normalized) ? 'success' : ['failed', 'error'].includes(normalized) ? 'danger' : 'pending'
  return <span className={`status ${tone}`}><span className="status-dot" />{categoryLabel(status)}</span>
}

function RunTable({ runs, selectedId, onSelect }: { runs: Run[]; selectedId: string | null; onSelect: (id: string) => void }) {
  if (runs.length === 0) return <EmptyState title="No evaluation runs yet">Create a dataset and launch an evaluation to see results here.</EmptyState>
  return <div className="table-scroll"><table><thead><tr><th>Model</th><th>Dataset</th><th>Tasks</th><th>Overall</th><th>Created</th><th><span className="sr-only">Open</span></th></tr></thead><tbody>
    {runs.map((run) => <tr key={run.id} className={selectedId === run.id ? 'selected-row' : undefined}>
      <td><strong>{run.model}</strong><div className="secondary cell-id">{run.id}</div></td>
      <td>{run.dataset_name}<span className="version-inline">v{run.dataset_version}</span></td>
      <td>{count(run.summary?.total_tasks)}</td>
      <td><span className="rate-cell">{pct(run.summary?.overall_success?.rate)}</span></td>
      <td className="secondary">{formatDate(run.created_at)}</td>
      <td><button type="button" className="table-link" onClick={() => onSelect(run.id)} aria-label={`View run ${run.id}`}>View →</button></td>
    </tr>)}
  </tbody></table></div>
}

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return <div className="trace-block"><div className="trace-label">{label}</div><pre>{pretty(value)}</pre></div>
}

function TracePanel({ result }: { result: TaskResult }) {
  const evaluation = result.evaluation
  const tokens = traceTokens(result)
  return <div className="trace-panel">
    <div className="trace-summary">
      <span className={isSuccess(result) ? 'result-good' : 'result-bad'}>{isSuccess(result) ? 'Passed' : 'Failed'}</span>
      {result.latency_ms != null && <span>{Math.round(result.latency_ms)} ms</span>}
      {tokens != null && <span>{count(tokens)} tokens</span>}
      <span>{result.turn?.tool_call?.name || 'No tool selected'}</span>
    </div>
    <div className="checks" aria-label="Metric outcomes">
      {[['Tool', evaluation.tool_selection_correct], ['Arguments', evaluation.arguments_correct], ['Schema', evaluation.schema_valid], ['Execution', evaluation.execution_success], ['Answer', evaluation.final_answer_correct]].map(([label, passed]) => <span key={String(label)} className={passed ? 'check good' : 'check bad'}>{passed ? '✓' : '×'} {label}</span>)}
    </div>
    <div className="trace-grid">
      <JsonBlock label="User input" value={result.task?.prompt ?? 'Input not available in this run'} />
      <JsonBlock label="Model output" value={result.turn?.content ?? result.turn?.raw ?? null} />
      <JsonBlock label="Selected tool" value={result.turn?.tool_call?.name ?? null} />
      <JsonBlock label="Tool arguments" value={result.turn?.tool_call?.arguments ?? null} />
      <JsonBlock label="Tool response" value={result.execution?.value ?? result.execution?.error ?? null} />
      <JsonBlock label="Final answer" value={result.final_turn?.content ?? result.turn?.content ?? null} />
    </div>
    {evaluation.details && <details className="raw-details"><summary>Expected vs actual</summary><JsonBlock label="Evaluation details" value={evaluation.details} /></details>}
    {Array.isArray(result.trace) && result.trace.length > 0 && <details className="raw-details"><summary>Full execution trace ({result.trace.length} events)</summary><JsonBlock label="Trace events, token usage and latency" value={result.trace} /></details>}
    {(evaluation.errors?.length || result.turn?.error || result.execution?.error || result.final_turn?.error) && <div className="trace-errors"><strong>Errors</strong><ul>{[...(evaluation.errors || []), result.turn?.error, result.execution?.error, result.final_turn?.error].filter(Boolean).map((error, index) => <li key={index}>{error}</li>)}</ul></div>}
  </div>
}

function RunDetailPanel({ detail, loading, error }: { detail: RunDetail | null; loading: boolean; error: string | null }) {
  const [taskFilter, setTaskFilter] = useState<'all' | 'failed'>('all')
  if (loading) return <div className="panel"><div className="loading">Loading run details…</div></div>
  if (error) return <ErrorBanner message={error} />
  if (!detail) return <EmptyState title="Select a run">Choose a run to inspect its metrics and execution trace.</EmptyState>
  const results = detail.report?.results || []
  const filtered = taskFilter === 'failed' ? results.filter((item) => !isSuccess(item)) : results
  return <section className="panel detail-panel" aria-label="Run details">
    <div className="section-heading"><div><div className="eyebrow">Run detail</div><h2>{detail.model}</h2><p>{detail.dataset_name} · v{detail.dataset_version} · {formatDate(detail.created_at)}</p></div><span className="run-id">{detail.id}</span></div>
    <div className="detail-metrics">{METRICS.map((item) => <div key={item.key}><span>{item.short}</span><strong>{pct(detail.summary?.[item.key]?.rate)}</strong></div>)}</div>
    <div className="subheading"><div><h3>Task traces</h3><span>{results.length} tasks</span></div><div className="segmented" role="group" aria-label="Filter tasks"><button type="button" className={taskFilter === 'all' ? 'active' : ''} onClick={() => setTaskFilter('all')}>All</button><button type="button" className={taskFilter === 'failed' ? 'active' : ''} onClick={() => setTaskFilter('failed')}>Failed</button></div></div>
    {filtered.length === 0 ? <EmptyState icon="✓" title={taskFilter === 'failed' ? 'No failed tasks' : 'No task traces'}>{taskFilter === 'failed' ? 'All tasks in this run passed the available checks.' : 'This run has no per-task results.'}</EmptyState> : <div className="task-list">{filtered.map((result, index) => <details className="task-item" key={`${taskId(result)}-${index}`}><summary><span className={isSuccess(result) ? 'task-marker good' : 'task-marker bad'}>{isSuccess(result) ? '✓' : '×'}</span><strong>{taskId(result)}</strong><span className="task-preview">{result.task?.prompt || result.turn?.tool_call?.name || 'Task details'}</span><span className="chevron">⌄</span></summary><TracePanel result={result} /></details>)}</div>}
  </section>
}

async function parseTaskFile(file: File): Promise<DatasetTask[]> {
  if (file.size > 10 * 1024 * 1024) throw new Error('Dataset files must be 10 MB or smaller.')
  const content = await file.text()
  let records: unknown
  try {
    if (file.name.toLowerCase().endsWith('.jsonl')) {
      records = content.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line, index) => {
        try { return JSON.parse(line) as unknown } catch { throw new Error(`Invalid JSON on non-empty line ${index + 1}.`) }
      })
    } else {
      const parsed = JSON.parse(content) as unknown
      records = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' && 'tasks' in parsed ? (parsed as { tasks: unknown }).tasks : parsed
    }
  } catch (error) {
    throw error instanceof Error ? error : new Error('Could not parse dataset file.')
  }
  if (!Array.isArray(records) || records.length === 0) throw new Error('Expected a non-empty JSON array, {"tasks": [...]}, or JSONL file.')
  records.forEach((record, index) => {
    if (!record || typeof record !== 'object' || Array.isArray(record)) throw new Error(`Task ${index + 1} is not an object.`)
    const task = record as Partial<DatasetTask>
    if (typeof task.task_id !== 'string' || typeof task.prompt !== 'string' || !Array.isArray(task.available_tools) || !('expected_tool' in task) || !task.expected_arguments || typeof task.expected_arguments !== 'object' || !('expected_answer' in task)) throw new Error(`Task ${index + 1} is missing required benchmark fields.`)
  })
  return records as DatasetTask[]
}

function ModelTable({ models }: { models: ModelMetrics[] }) {
  if (models.length === 0) return <EmptyState title="No models to compare">Results appear once an evaluation finishes.</EmptyState>
  return <div className="table-scroll"><table className="comparison-table"><thead><tr><th>Model</th><th>Runs</th><th>Tasks</th>{METRICS.map((item) => <th key={item.key}>{item.short}</th>)}</tr></thead><tbody>{models.map((model) => <tr key={model.model}><td><strong>{model.model}</strong></td><td>{count(model.runs)}</td><td>{count(model.total_tasks)}</td>{METRICS.map((item) => <td key={item.key}><span className={`comparison-score ${item.key === 'overall_success' ? 'prominent' : ''}`}>{pct(model.summary?.[item.key]?.rate)}</span></td>)}</tr>)}</tbody></table></div>
}

export default function App() {
  const [view, setView] = useState<View>('overview')
  const [stats, setStats] = useState<Stats | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [initialLoading, setInitialLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null)
  const [badcases, setBadcases] = useState<Badcase[]>([])
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [comparisonDatasetId, setComparisonDatasetId] = useState('')
  const [backendType, setBackendType] = useState<BackendConfig['type']>('mock')
  const [model, setModel] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [datasetVersion, setDatasetVersion] = useState('1.0.0')
  const [datasetDescription, setDatasetDescription] = useState('')
  const [datasetTasks, setDatasetTasks] = useState<DatasetTask[]>([])
  const [datasetFileName, setDatasetFileName] = useState('')
  const [datasetError, setDatasetError] = useState<string | null>(null)
  const [importing, setImporting] = useState(false)
  const [viewDataset, setViewDataset] = useState<Dataset | null>(null)
  const [datasetDetailError, setDatasetDetailError] = useState<string | null>(null)
  const [apiKeyOpen, setApiKeyOpen] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState(getApiKey)

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([api.stats(comparisonDatasetId || undefined), api.runs(), api.datasets()])
    const errors: string[] = []
    if (results[0].status === 'fulfilled') setStats(results[0].value)
    else errors.push(`Statistics: ${results[0].reason instanceof Error ? results[0].reason.message : 'unavailable'}`)
    if (results[1].status === 'fulfilled') {
      const loadedRuns = results[1].value
      setRuns(loadedRuns)
      setSelectedRunId((current) => current || loadedRuns[0]?.id || null)
    } else errors.push(`Runs: ${results[1].reason instanceof Error ? results[1].reason.message : 'unavailable'}`)
    if (results[2].status === 'fulfilled') {
      const loadedDatasets = results[2].value
      setDatasets(loadedDatasets)
      setSelectedDatasetId((current) => current || loadedDatasets[0]?.id || '')
    } else errors.push(`Datasets: ${results[2].reason instanceof Error ? results[2].reason.message : 'unavailable'}`)
    setLoadError(errors.length > 0 ? errors.join(' · ') : null)
    setInitialLoading(false)
  }, [comparisonDatasetId])

  useEffect(() => { void refresh() }, [refresh])

  useEffect(() => {
    if (!selectedRunId) { setRunDetail(null); setBadcases([]); return }
    let cancelled = false
    setDetailLoading(true)
    setDetailError(null)
    Promise.all([api.run(selectedRunId), api.badcases(selectedRunId)]).then(([detail, cases]) => {
      if (!cancelled) { setRunDetail(detail); setBadcases(cases) }
    }).catch((error: unknown) => {
      if (!cancelled) { setRunDetail(null); setBadcases([]); setDetailError(error instanceof Error ? error.message : 'Could not load run details.') }
    }).finally(() => { if (!cancelled) setDetailLoading(false) })
    return () => { cancelled = true }
  }, [selectedRunId])

  useEffect(() => {
    if (!job || !['queued', 'pending', 'running', 'started'].includes(job.status.toLowerCase())) return
    const timer = window.setInterval(() => {
      api.job(job.id).then((latest) => {
        setJob(latest)
        if (latest.run_id) { setSelectedRunId(latest.run_id); void refresh() }
      }).catch((error: unknown) => setJobError(error instanceof Error ? error.message : 'Could not check job status.'))
    }, 3000)
    return () => window.clearInterval(timer)
  }, [job, refresh])

  const models = useMemo(() => [...(stats?.model_metrics || [])].sort((a, b) => (b.summary?.overall_success?.rate || 0) - (a.summary?.overall_success?.rate || 0)), [stats])
  const latestRun = runs[0]
  const failureCounts = useMemo(() => badcases.reduce<Record<string, number>>((acc, badcase) => { acc[badcase.primary_category] = (acc[badcase.primary_category] || 0) + 1; return acc }, {}), [badcases])

  const submitEvaluation = async (event: FormEvent) => {
    event.preventDefault()
    if (!selectedDatasetId) return
    setSubmitting(true)
    setJobError(null)
    try {
      const backend: BackendConfig = { type: backendType }
      if (backendType !== 'mock' && model.trim()) backend.model = model.trim()
      const created = await api.createEvaluation(selectedDatasetId, backend)
      setJob(created)
      setView('runs')
      if (created.run_id) { setSelectedRunId(created.run_id); await refresh() }
    } catch (error) {
      setJobError(error instanceof Error ? error.message : 'Could not start evaluation.')
    } finally { setSubmitting(false) }
  }

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setDatasetError(null)
    try {
      const tasks = await parseTaskFile(file)
      setDatasetTasks(tasks)
      setDatasetFileName(file.name)
      if (!datasetName) setDatasetName(file.name.replace(/\.(jsonl|json)$/i, ''))
    } catch (error) {
      setDatasetTasks([])
      setDatasetFileName('')
      setDatasetError(error instanceof Error ? error.message : 'Could not read the dataset file.')
    }
  }

  const submitDataset = async (event: FormEvent) => {
    event.preventDefault()
    if (datasetTasks.length === 0) { setDatasetError('Choose a JSONL or JSON benchmark file first.'); return }
    setImporting(true)
    setDatasetError(null)
    try {
      const created = await api.createDataset({ name: datasetName.trim(), version: datasetVersion.trim(), description: datasetDescription.trim(), tasks: datasetTasks })
      setSelectedDatasetId(created.id)
      setDatasetTasks([]); setDatasetFileName(''); setDatasetName(''); setDatasetDescription('')
      await refresh()
      const detail = await api.dataset(created.id)
      setViewDataset(detail)
    } catch (error) {
      setDatasetError(error instanceof Error ? error.message : 'Could not import the dataset.')
    } finally { setImporting(false) }
  }

  const openDataset = async (id: string) => {
    setDatasetDetailError(null)
    setViewDataset(null)
    try { setViewDataset(await api.dataset(id)) }
    catch (error) { setDatasetDetailError(error instanceof Error ? error.message : 'Could not load dataset.') }
  }

  const exportDataset = async (dataset: Dataset) => {
    setDatasetDetailError(null)
    try { await api.exportDataset(dataset.id, dataset.name, dataset.version) }
    catch (error) { setDatasetDetailError(error instanceof Error ? error.message : 'Could not export dataset.') }
  }

  const saveApiKey = (event: FormEvent) => {
    event.preventDefault()
    setApiKey(apiKeyInput)
    setApiKeyOpen(false)
    void refresh()
  }

  return <div className="app-shell">
    <aside className="sidebar" aria-label="Main navigation">
      <div className="brand"><div className="brand-symbol">T<span>·</span></div><div><strong>ToolCall Bench</strong><span>Evaluation console</span></div></div>
      <nav>{NAV.map((item) => <button type="button" key={item.id} className={`nav-item ${view === item.id ? 'active' : ''}`} onClick={() => setView(item.id)} aria-current={view === item.id ? 'page' : undefined}><span className="nav-icon" aria-hidden="true">{item.icon}</span>{item.label}</button>)}</nav>
      <div className="sidebar-footer"><span className="live-dot" />API-connected dashboard<div className="sidebar-note">Metrics reflect stored evaluation runs.</div></div>
    </aside>

    <main className="main-content">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> {NAV.find((item) => item.id === view)?.label}</div><div className="top-actions"><button type="button" className="api-key-button" onClick={() => { setApiKeyInput(getApiKey()); setApiKeyOpen((current) => !current) }} aria-expanded={apiKeyOpen} aria-controls="api-key-panel">{getApiKey() ? '● API key' : 'API key'}</button><button type="button" className="icon-button" onClick={() => void refresh()} aria-label="Refresh dashboard" title="Refresh">↻</button><span className="top-avatar">TC</span></div></header>
      {apiKeyOpen && <form id="api-key-panel" className="api-key-panel" onSubmit={saveApiKey}><label>API key<input type="password" autoComplete="off" value={apiKeyInput} onChange={(event) => setApiKeyInput(event.target.value)} placeholder="X-API-Key, if enabled" /></label><p>Stored only for this browser tab session. Leave blank to remove it.</p><button type="submit" className="primary-button">Save key</button></form>}
      <div className="content">
        {loadError && <ErrorBanner message={loadError} retry={() => void refresh()} />}
        {view === 'comparison' && !initialLoading && <section className="panel"><label className="filter-select">Compare on one dataset version<select value={comparisonDatasetId} onChange={(event) => setComparisonDatasetId(event.target.value)}><option value="">Latest evaluated dataset</option>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · v{dataset.version}</option>)}</select></label><p className="form-hint">Showing only runs on {stats?.comparison_dataset ? `${stats.comparison_dataset.name} v${stats.comparison_dataset.version}` : 'the selected dataset version'}. Scores from different datasets are not pooled.</p></section>}
        {initialLoading ? <div className="loading-page" role="status"><div className="spinner" />Loading evaluation data…</div> : <>
          {view === 'overview' && <>
            <div className="page-heading"><div><div className="eyebrow">EVALUATION INTELLIGENCE</div><h1>Overview</h1><p>Performance and reliability across your agent evaluations.</p></div><button type="button" className="primary-button" onClick={() => setView('runs')}>+ New evaluation</button></div>
            <div className="kpi-grid"><div className="kpi-card"><span>Total runs</span><strong>{count(stats?.total_runs)}</strong><small>Completed evaluations</small></div><div className="kpi-card"><span>Evaluation jobs</span><strong>{count(stats?.total_jobs)}</strong><small>Submitted jobs</small></div><div className="kpi-card"><span>Failed jobs</span><strong>{count(stats?.failed_jobs)}</strong><small>Infrastructure or execution errors</small></div><div className="kpi-card accent"><span>Models evaluated</span><strong>{count(models.length)}</strong><small>Distinct model configurations</small></div></div>
            <div className="overview-grid"><section className="panel"><div className="section-heading"><div><h2>Latest run performance</h2><p>{latestRun ? `${latestRun.model} · ${latestRun.dataset_name} v${latestRun.dataset_version}` : 'No completed run available'}</p></div>{latestRun && <button type="button" className="text-button" onClick={() => { setSelectedRunId(latestRun.id); setView('runs') }}>View run →</button>}</div>{latestRun?.summary ? <div className="metric-list">{METRICS.map((item) => <MetricBar key={item.key} label={item.label} metric={latestRun.summary[item.key]} />)}</div> : <EmptyState title="No run metrics yet">Run an evaluation to populate this view.</EmptyState>}</section>
              <section className="panel"><div className="section-heading"><div><h2>Models at a glance</h2><p>Ranked by overall success</p></div><button type="button" className="text-button" onClick={() => setView('comparison')}>Compare all →</button></div>{models.length ? <div className="model-rank-list">{models.slice(0, 5).map((entry, index) => <div className="model-rank" key={entry.model}><span className="rank-index">{String(index + 1).padStart(2, '0')}</span><div><strong>{entry.model}</strong><small>{count(entry.runs)} runs · {count(entry.total_tasks)} tasks</small></div><span className="rank-score">{pct(entry.summary?.overall_success?.rate)}</span></div>)}</div> : <EmptyState title="No model results">Complete an evaluation to compare model performance.</EmptyState>}</section></div>
            <section className="panel"><div className="section-heading"><div><h2>Recent runs</h2><p>Open a run to inspect individual task traces.</p></div><button type="button" className="text-button" onClick={() => setView('runs')}>All runs →</button></div><RunTable runs={runs.slice(0, 5)} selectedId={null} onSelect={(id) => { setSelectedRunId(id); setView('runs') }} /></section>
          </>}

          {view === 'comparison' && <><div className="page-heading"><div><div className="eyebrow">BENCHMARK RESULTS</div><h1>Model comparison</h1><p>Aggregate scores across stored runs. Compare models on the same dataset for like-for-like conclusions.</p></div></div><section className="panel"><div className="section-heading"><div><h2>Cross-model metrics</h2><p>{models.length} model{models.length === 1 ? '' : 's'} · six evaluation dimensions</p></div></div><ModelTable models={models} /></section>{models.length > 0 && <div className="comparison-cards">{models.map((entry) => <section className="panel model-card" key={entry.model}><div className="model-card-head"><div><div className="eyebrow">MODEL</div><h2>{entry.model}</h2><span>{count(entry.runs)} runs · {count(entry.total_tasks)} tasks</span></div><strong>{pct(entry.summary?.overall_success?.rate)}</strong></div><div className="metric-list">{METRICS.slice(0, 5).map((item) => <MetricBar key={item.key} label={item.short} metric={entry.summary?.[item.key]} compact />)}</div></section>)}</div>}</>}

          {view === 'runs' && <><div className="page-heading"><div><div className="eyebrow">EVALUATION WORKFLOW</div><h1>Evaluation runs</h1><p>Start a benchmark, monitor its job, and inspect complete execution traces.</p></div></div><section className="panel launch-panel"><div className="section-heading"><div><h2>New evaluation</h2><p>API credentials and endpoint URLs are configured on the server.</p></div></div><form className="launch-form" onSubmit={(event) => void submitEvaluation(event)}><label>Dataset<select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)} required><option value="">Select a dataset</option>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · v{dataset.version} ({dataset.task_count} tasks)</option>)}</select></label><label>Backend<select value={backendType} onChange={(event) => setBackendType(event.target.value as BackendConfig['type'])}><option value="mock">Mock</option><option value="openai">OpenAI-compatible</option><option value="local">Local model</option></select></label><label>Model ID<input type="text" value={model} onChange={(event) => setModel(event.target.value)} disabled={backendType === 'mock'} placeholder={backendType === 'mock' ? 'Built-in mock agent' : 'Server-configured model ID'} /></label><button type="submit" className="primary-button" disabled={submitting || !selectedDatasetId}>{submitting ? 'Starting…' : 'Start evaluation'}</button></form>{datasets.length === 0 && <p className="form-hint">No dataset yet. Import one from the Datasets page to begin.</p>}{jobError && <ErrorBanner message={jobError} />}{job && <div className="job-banner" role="status"><div><strong>Job {job.id}</strong><span>Submitted {formatDate(job.created_at)}</span>{job.error && <span className="job-error">{job.error}</span>}</div><StatusPill status={job.status} /></div>}</section>
            <section className="panel"><div className="section-heading"><div><h2>Run history</h2><p>{runs.length} completed run{runs.length === 1 ? '' : 's'}</p></div></div><RunTable runs={runs} selectedId={selectedRunId} onSelect={setSelectedRunId} /></section><RunDetailPanel detail={runDetail} loading={detailLoading} error={detailError} /></>}

          {view === 'failures' && <><div className="page-heading"><div><div className="eyebrow">BADCASE ANALYSIS</div><h1>Failure analysis</h1><p>Classified task failures for a selected evaluation run.</p></div></div><section className="panel"><div className="section-heading"><div><h2>Run filter</h2><p>Failure taxonomy is generated by the evaluation engine.</p></div></div><label className="filter-select">Evaluation run<select value={selectedRunId || ''} onChange={(event) => setSelectedRunId(event.target.value || null)}><option value="">Select a run</option>{runs.map((run) => <option key={run.id} value={run.id}>{run.model} · {run.dataset_name} v{run.dataset_version} · {formatDate(run.created_at)}</option>)}</select></label></section>{detailLoading ? <div className="loading">Loading failure analysis…</div> : detailError ? <ErrorBanner message={detailError} /> : !selectedRunId ? <EmptyState title="Select a run">Choose an evaluation run to explore its failures.</EmptyState> : badcases.length === 0 ? <EmptyState icon="✓" title="No badcases in this run">All evaluated tasks passed, or no failures were classified.</EmptyState> : <><div className="failure-kpis"><div className="kpi-card"><span>Total badcases</span><strong>{count(badcases.length)}</strong><small>In selected run</small></div>{Object.entries(failureCounts).map(([category, total]) => <div className="kpi-card" key={category}><span>{categoryLabel(category)}</span><strong>{count(total)}</strong><small>{pct(total / badcases.length)} of badcases</small></div>)}</div><section className="panel"><div className="section-heading"><div><h2>Classified failures</h2><p>Inspect the classification and original task trace.</p></div></div><div className="badcase-list">{badcases.map((item, index) => <article className="badcase" key={`${item.task_id}-${index}`}><div className="badcase-head"><span className="failure-tag">{categoryLabel(item.primary_category)}</span><strong>{item.task_id}</strong><button type="button" className="text-button" onClick={() => setView('runs')}>View trace →</button></div><p>{item.reason || 'No reason provided.'}</p><details className="raw-details"><summary>Structured badcase data</summary><JsonBlock label="Failure record" value={item} /></details></article>)}</div></section></>}</>}

          {view === 'datasets' && <><div className="page-heading"><div><div className="eyebrow">BENCHMARK CATALOG</div><h1>Datasets</h1><p>Versioned benchmark data for reproducible model evaluations.</p></div></div><div className="dataset-grid"><section className="panel"><div className="section-heading"><div><h2>Import dataset</h2><p>Upload JSONL, a JSON array, or a JSON object with a tasks array.</p></div></div><form className="dataset-form" onSubmit={(event) => void submitDataset(event)}><label>Name<input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} placeholder="e.g. customer-support-tools" required /></label><div className="field-row"><label>Version<input value={datasetVersion} onChange={(event) => setDatasetVersion(event.target.value)} placeholder="1.0.0" required /></label><label>Benchmark file<input type="file" accept=".jsonl,.json,application/json" onChange={(event) => void handleFile(event)} aria-describedby="file-help" /></label></div><label>Description <span className="optional">(optional)</span><textarea rows={3} value={datasetDescription} onChange={(event) => setDatasetDescription(event.target.value)} placeholder="What does this benchmark measure?" /></label><p id="file-help" className="form-hint">Required fields: task_id, prompt, available_tools, expected_tool, expected_arguments, expected_answer.</p>{datasetFileName && <div className="file-ready">✓ {datasetFileName} · {datasetTasks.length} tasks ready</div>}{datasetError && <ErrorBanner message={datasetError} />}<button type="submit" className="primary-button" disabled={importing || datasetTasks.length === 0}>{importing ? 'Importing…' : 'Import dataset'}</button></form></section><section className="panel"><div className="section-heading"><div><h2>Dataset catalog</h2><p>{datasets.length} version{datasets.length === 1 ? '' : 's'} available</p></div></div>{datasets.length === 0 ? <EmptyState title="No datasets imported">Upload a benchmark to create the first version.</EmptyState> : <div className="dataset-list">{datasets.map((dataset) => <article key={dataset.id} className="dataset-item"><div className="dataset-icon">▤</div><div><strong>{dataset.name}</strong><p>{dataset.description || 'No description'}</p><span>v{dataset.version} · {count(dataset.task_count)} tasks · {formatDate(dataset.created_at)}</span></div><div className="dataset-actions"><button type="button" className="text-button" onClick={() => void openDataset(dataset.id)}>Inspect</button><button type="button" className="text-button" onClick={() => void exportDataset(dataset)}>Export</button></div></article>)}</div>}</section></div>{datasetDetailError && <ErrorBanner message={datasetDetailError} />}{viewDataset && <section className="panel"><div className="section-heading"><div><div className="eyebrow">DATASET VERSION</div><h2>{viewDataset.name} <span className="version-inline">v{viewDataset.version}</span></h2><p>{viewDataset.description || 'No description'} · {count(viewDataset.task_count)} tasks</p></div><button type="button" className="text-button" onClick={() => setViewDataset(null)}>Close</button></div><div className="dataset-preview">{(viewDataset.tasks || []).map((task) => <details key={task.task_id} className="task-item"><summary><strong>{task.task_id}</strong><span className="task-preview">{task.prompt}</span><span className="chevron">⌄</span></summary><div className="trace-grid"><JsonBlock label="Prompt" value={task.prompt} /><JsonBlock label="Available tools" value={task.available_tools} /><JsonBlock label="Expected tool" value={task.expected_tool} /><JsonBlock label="Expected arguments" value={task.expected_arguments} /><JsonBlock label="Expected answer" value={task.expected_answer} /></div></details>)}{!viewDataset.tasks?.length && <EmptyState title="No task preview available">This dataset version has no returned task records.</EmptyState>}</div></section>}</>}
        </>}
      </div>
    </main>
  </div>
}
