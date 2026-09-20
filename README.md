# ToolCall-Bench-Lite

[![CI](https://github.com/senxichong/toolcall-bench-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/senxichong/toolcall-bench-lite/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个可追踪的 LLM Agent 工具调用评测平台。相同数据集可用于确定性 Mock、OpenAI 兼容接口及本地模型；平台提供五项指标、完整模型／工具调用轨迹、结构化失败分析，以及可选的 FastAPI + React 仪表盘。服务部署使用 PostgreSQL 保存数据、Redis + Celery 执行异步评测。

**注意：**Mock 在示例数据集上的成绩仅用于验证评测链路，不代表真实大模型的能力。

## 快速开始：离线命令行

需要 Python 3.10 或更新版本。

```bash
python -m pip install -e ".[dev]"
toolcall-bench run benchmarks/demo.jsonl
python -m pytest
ruff check .
```

示例包含 25 道任务及 5 个确定性的本地工具，无需密钥或联网。结果写入 `runs/latest.json`、`runs/latest.md` 和 `runs/latest.badcases.json`。

评测 OpenAI 兼容的 Chat Completions 接口：

```bash
export TOOLCALL_BENCH_MODEL="your-model"
export TOOLCALL_BENCH_BASE_URL="https://your-endpoint.example/v1"
export TOOLCALL_BENCH_API_KEY="your-secret"
toolcall-bench run benchmarks/demo.jsonl --adapter openai-compatible --output runs/model-a.json
```

PowerShell 中使用 `$env:TOOLCALL_BENCH_MODEL = "your-model"` 设置环境变量，其他变量同理。密钥没有命令行参数，避免写入命令历史。本地推理先执行 `python -m pip install -e ".[local]"`，再使用 `--adapter local --model 模型标识或路径`。模型首次调用时才加载，输出质量取决于模型遵循工具调用 JSON 格式的能力。

## 数据集格式与版本管理

基准数据采用 UTF-8 JSONL，每行一条任务：

```json
{"task_id":"calc_001","prompt":"What is 25 * 16?","available_tools":["calculator"],"expected_tool":"calculator","expected_arguments":{"expression":"25 * 16"},"expected_answer":"400","answer_match":"exact"}
```

`expected_tool` 可为 `null`，表示正确行为是不调用工具。`answer_match` 支持 `auto`、`exact`、`json`、`numeric` 和 `contains`；可选 `metadata` 保存任务标签。加载器会拒绝格式错误、重复任务 ID 和不一致的标准答案。

```bash
toolcall-bench dataset import arithmetic benchmarks/demo.jsonl
toolcall-bench dataset list --name arithmetic
toolcall-bench dataset verify arithmetic --version v000001
toolcall-bench dataset export arithmetic exported.jsonl --version v000001
toolcall-bench dataset diff arithmetic v000001 v000002
toolcall-bench run --dataset-name arithmetic --dataset-version v000001
```

数据集默认位于 `datasets/`；用 `toolcall-bench dataset --root 路径 ...` 修改管理目录，评测时用 `toolcall-bench run --datasets-dir 路径 ...` 指定相同目录。导入生成带任务数量和 SHA-256 摘要的不可变版本；再次导入内容相同的数据会返回原版本。运行报告保留数据集名称、版本和摘要。导出支持 JSONL 与 JSON 数组。

## 评测架构

`LLMBackend.generate(messages, tools)` 是模型无关的接口。内置 `MockBackend`、`OpenAICompatibleAdapter` 与 `LocalModelBackend`；旧名称 `ModelAdapter` 和 `MockAgent` 继续兼容。后端只接收对话和允许使用的工具定义，不接收标准答案。

运行器解析模型输出，验证任务工具白名单和参数 Schema，调用 `ToolRegistry`，再执行评测。每道任务保存按时间排序的模型／工具事件：用户输入、输入消息、原始模型输出、所选工具、参数、工具响应、延迟、可用的 Token 用量及错误。单个模型或工具异常会被记录为任务失败，不会直接中断整批评测。

| 指标 | 通过条件 |
|---|---|
| 工具选择准确率 | 所选工具与标准工具相同；正确地不调用工具也算通过。 |
| 参数正确性 | 参数结构与标准参数匹配；算术表达式支持规范化比较。 |
| Schema 有效性 | 响应可解析，工具在白名单内，参数满足已支持的 JSON Schema 子集。 |
| 执行成功率 | 工具调用成功，或任务本就无需工具调用。 |
| 最终答案质量 | 答案满足任务配置的确定性匹配规则。 |

五项全部通过才计为整体成功。Badcase 分析会标记 `wrong_tool`、`wrong_arguments`、`tool_failure`、`reasoning_failure`，保存主要类别、其他适用类别、预期与实际结果及轨迹。这里的 `reasoning_failure` 表示可观察的答案错误，并不声称能够判断模型内部思维过程。

## API、任务队列与仪表盘

仅在本机体验 API，可使用 SQLite 与同步 Mock 演示模式：

```bash
python -m pip install -e ".[service]"
export TOOLCALL_DEMO_MODE=true
uvicorn toolcall_bench.service.app:app --reload
```

API 文档位于 `http://localhost:8000/docs`。单独开发 React 仪表盘需要 Node.js 20.19+：

```bash
cd dashboard
npm ci
npm run dev
```

部署 PostgreSQL、Redis、Celery 和仪表盘时，先参照 `.env.example` 在 `.env` 中设置强随机 `POSTGRES_PASSWORD`，再运行：

```bash
docker compose up --build -d
```

默认本机地址为仪表盘 `http://localhost:3000`、API `http://localhost:8000/api`、接口文档 `http://localhost:8000/docs`。模型凭据应通过服务端环境变量 `TOOLCALL_BENCH_*` 配置，不应放入评测请求。默认 Docker 镜像不包含本地大模型推理的 `transformers` / `torch` 依赖；需要自建含 `[local]` 依赖的 Worker 镜像才能在容器内使用本地后端。

API 提供数据集创建／导入／列表／详情／导出、评测提交与任务状态、运行详情与 badcase、统计及对比。PostgreSQL 保存不可变的数据集 `(名称, 版本)` 快照与完整运行报告；普通模式通过 Redis 将任务交给 Celery Worker。仪表盘展示模型对比、各项指标、执行轨迹和失败类别。

不要直接将默认演示部署暴露到公网。对外服务应配置 `TOOLCALL_API_KEY` 或上游身份认证、TLS、访问控制、备份和监控。轨迹包含任务提示词与工具输出，可能含敏感信息。详见[架构与运行边界](docs/architecture.md)。

## 扩展与测试

```python
from toolcall_bench import LLMBackend, BenchmarkRunner

class MyBackend(LLMBackend):
    name = "my-backend"

    def generate(self, messages, tools):
        return my_client.complete(
            messages=messages,
            tools=[tool.to_openai_tool() for tool in tools],
        )

report = BenchmarkRunner(adapter=MyBackend()).run("benchmarks/demo.jsonl")
```

增加工具可参考[自定义工具示例](examples/custom_tool.py)。提交前运行 `python -m pytest`、`ruff check .`，并在 `dashboard/` 中运行 `npm ci && npm run build`。CI 会检查 Python 代码、测试和前端构建。

当前边界：每道题最多执行一次工具调用；只支持实用的 JSON Schema 子集；最终答案采用确定性比较而非语义评审；每题只有一组标准工具调用；尚无多租户权限、数据库迁移和数据保留策略。这些限制不应被视为已达到完整企业生产环境要求。

项目采用 [MIT 许可证](LICENSE)。
