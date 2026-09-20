# Architecture and operating model

ToolCall-Bench-Lite separates benchmark data, model generation, deterministic evaluation, and the optional web service. The Python runner remains usable without a database or network connection. The service wraps the same runner rather than duplicating scoring logic.

```text
JSONL / versioned dataset / API dataset snapshot
                  │
                  ▼
             BenchmarkRunner ──► LLMBackend
                  │                  ├─ mock
                  │                  ├─ OpenAI-compatible chat completions
                  │                  └─ local Transformers model
                  │
                  ├─► parser ─► allowlist + JSON Schema subset ─► ToolRegistry
                  ├─► ordered model/tool trace with latency and token usage
                  ├─► five-metric evaluator
                  └─► JSON/Markdown report ─► rule-based badcase analysis

React dashboard ──► FastAPI ──► PostgreSQL: dataset snapshots, jobs, runs
                         │
                         └──► Redis/Celery ──► worker ──► BenchmarkRunner
```

## Core contracts

`LLMBackend.generate(messages, tools)` accepts only the conversation and allowed tool specifications. The runner does not branch on the provider. Adapter output is normalized by the parser; malformed output is recorded as a failed evaluation instead of terminating a batch. The mock backend is an offline regression fixture, not a representative language model.

Each task has one expected tool call, or no tool call. The runner supports one tool invocation and a subsequent answer turn. It records ordered model/tool trace events containing input messages, raw model output, selected tool, arguments, tool response, latency, reported token usage, and error information. Token usage is nullable when a backend does not report it. Provider error bodies and arbitrary tool exception strings are not persisted; common credential-shaped strings are redacted during report serialization. Traces may still contain sensitive *task prompts and tool outputs*: operators must govern dataset and report access accordingly.

The five per-task checks are tool selection accuracy, argument correctness, schema validity, execution success, and final-answer accuracy. Their conjunction is overall success. Final-answer scoring uses the task's deterministic `answer_match` policy; it is not a semantic judge. The badcase analyzer assigns one primary category and can retain multiple applicable categories: `wrong_tool`, `wrong_arguments`, `tool_failure`, and `reasoning_failure`. These are evidence-based labels, not a causal diagnosis of a model's internal reasoning.

## Dataset identity

The CLI `DatasetManager` imports validated JSONL or JSON arrays into `datasets/<name>/vNNNNNN/`, with a manifest and canonical task payload. Versions are immutable by convention and verified against a SHA-256 digest and task count before use or export. Re-importing identical canonical content returns the existing version. A run records the dataset name, version, and digest. `dataset diff` compares task IDs and task contents.

The API stores a submitted `(name, version)` task snapshot in PostgreSQL and rejects duplicates. Jobs reference that snapshot; completed runs retain the dataset identity and full report. API dataset IDs/versions and CLI dataset manifests are separate storage interfaces, so import/export JSONL when moving between them.

## Service lifecycle

1. A client creates or imports a dataset snapshot through FastAPI.
2. It submits an evaluation with a credential-free backend configuration. Credentials and endpoint URL come from worker environment variables, not the request body.
3. FastAPI commits a queued job. In normal mode it enqueues the job to Redis for a Celery worker; explicit demo mode runs mock jobs in-process.
4. The worker loads the snapshot, runs the core evaluator, and commits a run report and terminal job state.
5. The dashboard polls job state, then queries run details, aggregate statistics, model comparisons, and structured badcases.

PostgreSQL holds durable state. Redis is a task broker, not the system of record. SQLite can be used for local tests/demo; use PostgreSQL for a deployed multi-process service. The schema is created on startup. For long-lived deployments, introduce migrations before making incompatible schema changes; `create_all` is not a migration system.

## Deployment and trust boundaries

Docker Compose runs PostgreSQL, Redis, API, worker, and an Nginx-served React build. Configure a strong PostgreSQL password and, when exposing the service beyond a trusted local network, an API key or upstream authentication/TLS. Restrict who can submit datasets and jobs: prompts, model output, and tool results may be confidential. The current API has a single optional shared key, no per-user roles, tenant isolation, quotas, or retention policy. It should be placed behind an authenticated reverse proxy for multi-user use.

The service allows only registered local tools. The local-model backend may download model artifacts at first use; install optional dependencies and pin/trust model sources separately. Do not register side-effectful tools without an isolation and permission policy. The OpenAI-compatible endpoint is set by the operator environment; do not expose arbitrary endpoint URLs from untrusted API requests.

## Extension points and current limits

- Implement `LLMBackend` for another model provider; keep provider-specific parsing inside that adapter or the parser.
- Add tools through `ToolRegistry` with explicit JSON Schema parameters.
- Add grading policies or multiple acceptable answers without changing the storage/API layers.
- Current evaluation is one tool call per task, not a general multi-step agent trajectory. Parallel calls, semantic answer judging, provider retries, distributed rate limits, benchmark access control, and schema migrations remain future work.
