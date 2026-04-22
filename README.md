# easyopenai

异步多 provider 聚合调度的 OpenAI 兼容客户端。一份配置文件，自动在多家 API 提供商之间分发任务、处理流式/非流式差异、解析思考模型的 reasoning 内容、统计 token、限流重试与故障切换。

## 特性

- **多 provider 聚合调度**：每个 provider 独立配置 base url / api key / 并发数 / RPM；任务从共享队列动态分发，谁空闲谁接
- **流式 → 非流式透明化**：`force_stream: true` 让只支持 stream 的 provider 在内部聚合为一次性响应，对外接口完全一致
- **思考模型兼容**：自动从 `reasoning_content` / `reasoning` / `<think>...</think>` 三种格式中提取思考内容，与最终回答分离
- **健壮性**：指数退避重试 5xx/超时；滑动窗口失败率熔断（CLOSED → OPEN → HALF\_OPEN），故障 provider 上的任务自动回流到其他 provider
- **Token 统计**：分别统计 prompt / completion / reasoning / answer 四类 token
- **Key 安全**：YAML 中用 `${ENV_VAR}` 插值，配合 `.env`；日志自动脱敏
- **极简 API**：`async for result in client.stream(tasks)` 一行搞定

## 安装

### 作为依赖使用

```bash
# uv (推荐)
uv add "git+https://github.com/jubgjf/easyopenai.git"

# pip
pip install "git+https://github.com/jubgjf/easyopenai.git"
```

在 `pyproject.toml` 里声明：

```toml
dependencies = [
    "easyopenai @ git+https://github.com/jubgjf/easyopenai.git",
]
```

锁定到某个 tag / commit：

```bash
uv add "git+https://github.com/jubgjf/easyopenai.git@v0.1.1"
```

### 本地开发

```bash
git clone https://github.com/jubgjf/easyopenai.git
cd easyopenai
uv sync
```

需要 Python 3.12+。

## 快速开始

**1. 准备配置**

```bash
cp config.example.yaml config.yaml
cp .env.example .env
# 编辑 .env 填入真实 API key
```

**2. 配置文件 `config.yaml`**

```yaml
logging:
  stats_interval_s: 30        # 每 30 秒打印 provider 吞吐统计
scheduler:
  max_retries_per_task: 3     # 一个任务最多被多少 provider 尝试
  dispatch_policy: round_robin  # greedy(默认): 谁空闲谁接, 吞吐优先
                                # round_robin: 轮流分发, 各 provider 均匀扣费
providers:
  - name: bltcy
    base_url: https://api.bltcy.ai/v1
    api_key: ${BLTCY_API_KEY}      # 从环境变量解析
    max_concurrency: 4              # 并发请求数
    max_rpm: 60                     # 每分钟最大请求数
    force_stream: true              # 该 provider 仅支持流式则设 true
    health:
      window_size: 20               # 滑动窗口大小
      failure_threshold: 0.5        # 失败率阈值，触发熔断
      cooldown_s: 30                # 熔断后多久进入半开探测
    models:
      - name: qwen3-8b
        is_reasoning: true          # 思考模型
      - name: deepseek-r1
        is_reasoning: true

  - name: yunwu
    base_url: https://yunwu.ai/v1
    api_key: ${YUNWU_API_KEY}
    max_concurrency: 4
    max_rpm: 60
    force_stream: false
    models:
      - name: deepseek-r1
        is_reasoning: true
```

**工作方式**：每个 provider 启动 `max_concurrency` 个 worker 从共享队列拉任务，只处理自己支持且未尝试过的任务。任务在一个 provider 上失败时，自动重排到**还没尝试过**的其他 provider；全部 provider 都失败才返回带 `error` 字段的 `Result`。频繁失败的 provider 会被熔断跳过，冷却后自动探测恢复。

**调度策略 `scheduler.dispatch_policy`**：

- `greedy`（默认）：所有 worker 抢一个共享队列。延迟最低、吞吐最高，但 provider 列表里靠前的会被优先打满 —— 第一个 provider 没占满 `max_concurrency` 之前，后面的几乎拿不到任务。
- `round_robin`：dispatcher 协程把任务按顺序轮流派给各 provider 的私有队列。各家请求量均衡，适合**多 API 平台均匀扣费、规避单家突发风控**等场景。代价是有 ~5% 左右的吞吐折损（多了一层调度），实测可忽略。

**3. 完整示例：库的所有公开用法**

```python
"""easyopenai 所有对外接口一网打尽。

公开 API 仅包含: Client, Task, Result, TokenUsage, setup_logging
"""
import asyncio

from easyopenai import Client, Result, Task, TokenUsage, setup_logging


# --- 可选: 自定义日志级别 / 落盘 --------------------------------------------
# 不调用也可以, Client() 内部会以 INFO 级别自动初始化.
# setup_logging() 是幂等的, 重复调用只有第一次生效.
setup_logging(level="DEBUG", file="easyopenai.log")


# --- 用法 1: 同步单条调用, 最简形态 ------------------------------------------
# Client.chat() 自己管理事件循环, 适合脚本里一问一答.
# 注意: 不要在已有 asyncio 循环里调用 chat() —— 用 achat() 或 stream().
client = Client(config_path="config.yaml")
r: Result = client.chat(
    messages=[{"role": "user", "content": "1+1=?"}],
    model="deepseek-r1",
    max_tokens=100,
)
print(r.answer_content)          # 最终回答
print(r.reasoning_content)       # 思考内容 (非 reasoning 模型为空串)
print(r.usage)                   # TokenUsage(prompt=…, completion=…, reasoning=…, answer=…)
print(r.provider, r.latency_s, r.ok, r.error)


# --- 用法 2: 异步上下文 + 批量流式产出 (主流用法) ----------------------------
async def batch_example():
    tasks: list[Task | dict] = [
        # 方式 A: 直接构造 Task 对象
        Task(
            task_id="q-typed",
            messages=[{"role": "user", "content": "用一句话介绍数字 7"}],
            model="deepseek-r1",
            temperature=0.7,
            max_tokens=300,
            extra={"top_p": 0.9},     # 任意 OpenAI SDK 支持的字段, 透传
        ),
        # 方式 B: 传 dict, Client 会自动构造 Task
        {
            "messages": [{"role": "user", "content": "用一句话介绍数字 42"}],
            "model": "qwen3-8b",
            "max_tokens": 300,
        },
    ]

    # Client 必须以 async with 形式使用: 进入时探活所有 provider 并启动统计日志.
    async with Client(config_path="config.yaml", log_level="INFO") as client:
        # stream() 按完成顺序异步产出 Result (不保证与提交顺序一致)
        async for result in client.stream(tasks):
            if result.ok:
                print(f"[{result.task_id}] via {result.provider} ({result.latency_s:.2f}s)")
                print(f"  thinking: {len(result.reasoning_content)} chars")
                print(f"  answer:   {result.answer_content}")

                u: TokenUsage = result.usage
                print(f"  tokens:   prompt={u.prompt_tokens} "
                      f"completion={u.completion_tokens} "
                      f"reasoning={u.reasoning_tokens} "
                      f"answer={u.answer_tokens}")
            else:
                # 所有支持该 model 的 provider 都失败时到达这里
                print(f"[{result.task_id}] FAILED: {result.error}")


# --- 用法 3: 在已有事件循环中单条调用 ----------------------------------------
async def single_async_example():
    async with Client(config_path="config.yaml") as client:
        r = await client.achat(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-r1",
            max_tokens=50,
        )
        return r


if __name__ == "__main__":
    asyncio.run(batch_example())
    # result = asyncio.run(single_async_example())
```

运行：

```bash
uv run python your_script.py
```

## API 参考

### `Client`

```python
Client(config_path: str | Path | None = None,
       config: AppConfig | None = None,
       log_level: str = "INFO")
```

二选一传入 `config_path`（YAML 路径）或 `config`（已构造的 `AppConfig`）。必须以 `async with` 形式使用 —— 进入时会对所有 provider 探活，并启动周期统计日志协程。

#### `client.stream(tasks)`

```python
async def stream(tasks: Iterable[Task | dict]) -> AsyncIterator[Result]
```

主接口。提交一批任务，按完成顺序异步产出 `Result`。任务可传 `Task` 实例或可构造 `Task` 的 dict。

#### `client.achat(messages, model, **kw)` / `client.chat(...)`

单条便利方法。`achat` 是协程；`chat` 是同步包装（自带 `asyncio.run`，会创建独立的 Client 上下文，**不要在已有事件循环里调用**）。

### `Task`

```python
Task(task_id: str = <uuid>,
     messages: list[dict],
     model: str,
     temperature: float | None = None,
     max_tokens: int | None = None,
     extra: dict = {})        # 透传给 OpenAI SDK 的额外字段
```

### `Result`

```python
Result(task_id: str,
       provider: str,               # 最终服务该任务的 provider 名
       model: str,
       reasoning_content: str,      # 思考内容（非 reasoning 模型为空串）
       answer_content: str,         # 最终回答
       usage: TokenUsage,
       latency_s: float,
       error: str | None)           # None 表示成功

result.ok  # == (error is None)
```

### `TokenUsage`

```python
TokenUsage(prompt_tokens, completion_tokens, reasoning_tokens, answer_tokens)
# answer_tokens = completion_tokens - reasoning_tokens
```

## 开发 / 贡献

见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。
