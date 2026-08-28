# ToolCall-Bench-Lite 面试问答

## 1. Tool Calling 是什么？

Tool Calling（也常叫 Function Calling）是一种“模型负责决策、宿主程序负责执行”的协作协议。应用把工具名称、用途和参数 schema 交给模型；模型遇到需要外部能力的任务时，不直接假装得到结果，而是输出结构化调用，例如：

```json
{"name":"calculator","arguments":{"expression":"25 * 16"}}
```

宿主程序解析并校验这次调用，真正执行函数，再把 tool result 放回对话。模型最后基于真实结果回答用户。因此一次完整链路至少包括：选工具、生成参数、执行工具、消费结果、生成最终回答。

关键边界是：LLM 只“提出调用”，并没有直接执行 Python 函数或访问数据库。执行权限、错误处理和审计都属于宿主系统。本项目用 `ModelAdapter → parser → ToolRegistry → ModelAdapter` 把这条边界显式化。

## 2. Tool Call 为什么需要 schema？

Schema 是模型与执行器之间可机器验证的契约，至少解决四类问题：

1. **可发现性**：模型知道有哪些字段、字段含义和可选值。
2. **确定性解析**：执行器不需要从自然语言猜参数。
3. **错误隔离**：缺字段、类型错误、额外字段可在函数运行前失败。
4. **安全边界**：结合每次任务的 tool allowlist，可以阻止模型调用未授权工具或塞入未声明参数。

例如温度工具把单位约束为 `celsius | fahrenheit | kelvin`，就能把 `"C"`、`"centigrade"` 等不一致输出暴露为 schema failure，而不是把含糊值传到业务代码。

不过 schema 不是完整安全方案。它不能判断“用户是否真的授权付款”，也不能抵御工具内部的 SQL 注入或越权逻辑。真实系统还需要最小权限、输入消毒、超时/配额、人工确认和审计。

## 3. 如何评估 Argument Accuracy？

一个实用评估管线可以分四层：

1. **可解析性**：arguments 是否为合法 JSON object。
2. **schema 合法性**：类型、required、enum、边界和额外字段是否正确。
3. **字段级正确性**：逐字段比较，统计 micro/macro accuracy，便于定位问题。
4. **任务级正确性**：所有必要字段整体正确，才计一次 Argument Match 成功。

比较时应按字段语义做有限规范化：数字使用小容差；枚举统一大小写的前提是 schema 也允许；日期、单位和 ID 使用领域 canonicalizer；嵌套对象递归比较。表达式可比较 AST，而不是比较空格。不能随意把字符串 `"1"` 当成数字 `1`，因为这会隐藏真实类型错误。

本项目选择严格的 task-level Boolean 指标，同时对数字使用 `isclose`、对算术表达式比较 AST、对对象要求相同 key set。报告保留 expected/actual arguments，后续可以自然扩展为字段级 partial credit。

## 4. Exact Match 有什么问题？

Exact Match 简单、便宜、可复现，但容易产生假阴性：

- JSON key 顺序不同但语义相同；
- `25*16` 与 `25 * 16`；
- `32` 与 `32.0`；
- 自然语言中的大小写、空格或同义表达不同；
- 同一个目标存在多条合法工具路径；
- 最终答案多一句解释，虽然事实正确却无法逐字匹配。

反过来，过度宽松也会产生假阳性。例如只检查答案中是否出现 `400`，可能放过“结果不是 400”。因此需要按数据类型选择 grader：JSON 做结构比较、数值做容差比较、ID/枚举做严格比较、自由文本可用规则或有来源记录的语义 judge。

本项目让每条 task 选择 `exact/json/numeric/contains/auto`。它保持评分透明，但没有声称解决开放式语义评估。

## 5. 如何处理多步 Tool Calling？

多步场景不能只评估一次函数名。应把 Agent 看成受预算约束的状态机：

```text
用户任务 → 调用 A → 结果 A → 调用 B → 结果 B → 最终回答/终止
```

评估通常包含：

- 每一步工具和参数是否合法；
- 前一步结果是否被正确用于后一步；
- 是否遗漏必要步骤、重复调用或过早停止；
- 总步数、延迟、token 和工具成本；
- 最终环境状态与答案是否正确；
- 是否在最大步数内正常终止。

只做 gold trajectory exact match 会惩罚“路径不同但结果正确”的方案。更合理的是同时提供必需约束（必须取得哪些证据）、禁止约束（不能调用哪些工具）和最终状态断言；必要时允许多条参考轨迹。

ToolCall-Bench-Lite 当前刻意只执行一步。扩展时可以把 runner 改为循环，记录 `list[AgentTurn, ToolExecutionResult]`，增加 step budget、cycle detection、并行 call group 和 trajectory-level evaluator，而无需改变 ModelAdapter 的单轮接口。

## 6. 如何处理 Tool Error？

首先要区分错误来源：

- **模型错误**：选错工具、参数缺失、schema 不合法；
- **业务错误**：商品不存在、余额不足、输入超出领域范围；
- **基础设施错误**：超时、限流、网络中断、服务 5xx；
- **系统错误**：工具实现 bug 或返回格式损坏。

错误应作为结构化结果返回，例如 `success=false, error_type, retryable, message`，而不是让整个 benchmark 进程崩溃。Agent 可以据此修正参数、换工具、请求用户补充信息或清楚说明失败。

重试必须有上限和退避策略；有副作用的工具还需要 idempotency key，避免超时后重复扣款。评测数据应标记预期错误，分别统计“工具本身执行成功率”和“Agent 面对错误后的恢复率”。

本项目把本地异常捕获为 `ToolExecutionResult(success=False)`，仍把结果返回给 Adapter，并把错误写进 JSON report。当前 demo 不把故意失败伪装成成功；多步自动恢复属于 roadmap。

## 7. 如何防止错误 Tool Invocation？

防护应放在执行器侧，而不是依赖模型“听话”：

1. 每次请求只暴露最小 tool allowlist；
2. 严格解析 JSON，并在执行前验证 schema；
3. 拒绝额外字段、未知工具和越界值；
4. 为工具设置超时、速率、调用次数和资源预算；
5. 文件、网络、数据库凭证使用最小权限和隔离环境；
6. 付款、删除、发消息等高风险动作要求用户确认；
7. 区分用户指令、网页内容和 tool result 的来源，防止 prompt injection 把不可信内容升级为授权；
8. 记录工具名、参数、授权依据、结果和错误，支持审计与回放。

ToolCall-Bench-Lite 已实现前三项的轻量版本：任务级 `available_tools`、JSON/schema 校验和禁止执行未暴露工具。内置 calculator 使用 AST 白名单而不是 `eval`。

## 8. 这个项目如何扩展到真实 Agent？

可以保持当前模块边界，逐层替换能力：

- **Tools**：给 ToolSpec 增加权限等级、副作用、超时和幂等属性；registry 支持 async、HTTP、数据库和沙箱执行器。
- **Adapter**：增加不同厂商、流式响应和原生 Responses/Agents API 适配，同时保留原始响应便于审计。
- **Runner**：从单步变成带预算的多步循环，支持并行 calls、重试、人工确认和 checkpoint。
- **Dataset**：引入版本、标签、难度、领域、多个可接受答案和环境快照。
- **Evaluator**：增加字段级 partial credit、trajectory constraint、最终环境状态验证和可选语义 judge；所有 judge 都记录模型版本与 prompt。
- **Observability**：记录 token、延迟、成本、错误类型、调用 DAG，并支持 deterministic replay。
- **CI/Release**：用固定数据集做回归门禁，把真实模型结果和离线 Mock 测试分开发布。

真实 Agent 的核心不是“接更多 API”，而是把决策、权限、执行、状态和评估都变成可观察、可验证的边界。本项目提供的价值正是一个足够小、可以读懂并继续演进的骨架。

