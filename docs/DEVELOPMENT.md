# 开发者文档

面向贡献者与二次开发者。用户使用文档见根目录 [README.md](../README.md)。

## 技术栈与基本约定

- Python 3.12，`uv` 管理依赖与构建
- `asyncio` 全程；`pytest-asyncio` 以 `auto` 模式运行（见 `pyproject.toml`），直接 `async def test_*` 即可
- pydantic v2 做所有数据模型
- **日志统一用 loguru**，禁用标准库 `logging`。入口统一走 `easyopenai.logging.setup_logging()`，它注入了一个正则过滤器自动 mask `sk-…` / `Bearer …`
- `assert` 用于 fail-fast 不变量，不要软化为 warning / raise，除非有充分理由

## 目录结构

```
src/easyopenai/
  __init__.py      # 公开 API: 仅导出 Client, Task, Result, TokenUsage, setup_logging
  client.py        # 门面: async with 管理探活 / 统计协程 / provider 清理
  scheduler.py     # 任务队列 + worker 协程 + 失败回流
  provider.py      # AsyncOpenAI 封装: 限流 / 重试 / 健康记录 / stream 聚合
  stream.py        # 流式 chunk → 统一响应 dict 聚合
  parser.py        # reasoning 多形态解析 + fail-fast assert
  health.py        # 滑动窗口熔断三态机
  stats.py         # 周期性吞吐日志协程
  config.py        # YAML + ${ENV} 插值 + pydantic schema
  types.py         # Task / Result / TokenUsage / ProviderStats
  logging.py       # loguru 配置 + key 脱敏过滤器

tests/
  test_config.py             # ${VAR} 插值、缺失 env fail fast、key mask
  test_parser.py             # 三种 reasoning 形态 + <think> 兜底 + fail fast
  test_stream.py             # 流式 chunk 聚合、reasoning/content 分离
  test_health.py             # 熔断三态机所有迁移路径
  test_fault_injection.py    # respx mock: 故障切换、全失败、熔断打开、tenacity 重试
  test_smoke_real_api.py     # 真实 API 端到端（默认 skip）
```

## 请求生命周期

```
Client.stream(tasks)
  └─ 每次调用 new 一个 Scheduler
      └─ 为每个 Provider 起 max_concurrency 个 worker 协程
         worker 循环:
           task = await task_q.get()
           if not provider.supports(task.model)
              or provider.name in task.attempted_providers
              or not provider.health.can_serve():
               await task_q.put(task); sleep(0.05); continue
           task.attempted_providers.add(provider.name)
           try:
               result = await provider.call(task)     # 内部: aiolimiter → tenacity → (stream?) → parser
               result_q.put(result)
           except ProviderError:
               untried = [p for p in providers if p.supports(model) and p.name not in attempted]
               if untried and retry_count <= max_retries_per_task:
                   task_q.put(task)                    # 回流
               else:
                   result_q.put(Result(error=...))    # 彻底失败
```

`Client.stream` 是异步生成器，按完成顺序 yield `Result`，总数等于输入任务数（失败也会产出一条带 `error` 的）。

## 关键不变量（改代码前务必理解）

1. **失败回流排他性**：`task.attempted_providers` 保证同一任务不会被回流给刚失败过的 provider。改 scheduler 时别忘了在 `call()` 之前把 provider 加进去，否则 `ProviderError` 分支 `untried` 永远包含自己。

2. **流式/非流式对称**：`Provider._single_call` 按 `cfg.force_stream` 分支，但两条路径最终都返回 `dict` 结构的 "chat completion"（流式走 `aggregate_stream()`，非流式走 `resp.model_dump()`）。后续 `parser.parse_message` 只认 dict。新增 provider 行为时保持这个契约。

3. **reasoning parser fail-fast**：`parse_message(msg, is_reasoning=True)` 在三种提取方式都失败时 `assert False`。这是 README 的硬承诺 —— 不要改成 warning。

4. **熔断三态机** (`health.py`)：
   - `CLOSED → OPEN`：窗口填满且失败率 ≥ 阈值
   - `OPEN → HALF_OPEN`：`can_serve()` 检查 cooldown 到期，同时**立即**把 `_half_open_in_flight = True`（只放行 1 个探测）
   - `HALF_OPEN → CLOSED / OPEN`：下一次 `record()` 根据成败决定；记得清 `_half_open_in_flight`
   
   这里 in-flight 标志位置曾经出过 bug（见 `test_half_open_after_cooldown_and_recover`），改时多跑单测。

5. **token 会计**：`reasoning_tokens` 来自 `usage.completion_tokens_details.reasoning_tokens`（两家 provider 都是这个字段，见 `example.txt` 参考响应），`answer_tokens = completion_tokens - reasoning_tokens`，下限 clamp 到 0。

## 常用命令

```bash
uv sync                                   # 安装依赖（含 dev 组）
uv run pytest -v                          # 全量离线单测（含 respx 故障注入，~30s）
uv run pytest tests/test_health.py -v     # 单文件
uv run pytest -k reasoning -v             # 按名筛选

# 真实 API 冒烟测试（仅 2 次付费调用）：
BLTCY_API_KEY=... YUNWU_API_KEY=... \
  uv run pytest tests/test_smoke_real_api.py -v -s
```

**不要**在 `test_smoke_real_api.py` 里加大量 task —— 真实 API 调用花钱，目前限制 2 次/run 足以覆盖流式/非流式两条路径。

## 添加新 Provider 后端

绝大多数 OpenAI 兼容后端不用改代码，加一条 YAML 配置即可。仅当后端返回的字段路径超出以下 3 种时，才需要改 `parser.py`：

- `message.reasoning_content`（DeepSeek / Qwen 系列 / SiliconFlow）
- `message.reasoning`（OpenRouter 等）
- `<think>…</think>` 内联（部分本地后端）

如果某后端 usage 字段结构不同，改 `Provider._parse_usage` 即可。

## 添加新的故障场景测试

在 `tests/test_fault_injection.py` 中用 respx 拦截 `base_url` 路由。注意：

- 两个 provider 的 `.get("/models")` 路由都要 mock，否则 `Client.__aenter__` 的探活会真实联网失败
- 不要设 `respx.mock(assert_all_called=True)` —— 熔断场景里某些路由可能被故意不触发
- tenacity 默认 3 次重试 + 指数退避（`wait_random_exponential(min=1, max=10)`），单个 5xx 测试可能跑 10+ 秒，属正常

## 首版不支持 (YAGNI)

embedding / vision / tool calling、持久化任务队列、provider 优先级权重、动态配置热更新、分布式调度。如需扩展，优先考虑保持现有模块边界，例如 tool calling 只涉及 `Task.extra` 透传 + `Result` 新增字段，不需要改 scheduler。
