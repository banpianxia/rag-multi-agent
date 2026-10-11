# 多智能体 RAG 服务 & 邮件 HITL Agent —— 接口说明

> 本版已合并"修改版"的全部改动（5 个高危 bug + 14 个中低危 bug + 6 项优化），完整清单见文末「附录 A · 修改清单」。
> Web 服务端口为 **8008**。

本目录包含两个可独立运行的程序，均为去注释精简版。

| 文件 | 作用 | 运行方式 |
|---|---|---|
| `multi_agent_rag_service.py` | 个人知识库多智能体问答服务（**单文件版**，已内联原 `ingest.py` 的多格式摄入模块）：LangGraph 状态机 + 5 个 Agent 协作 + 三层记忆 + 人机协同 + LLM-as-judge 评测 + 多格式上传入库 + FastAPI 服务化 | 交互 / HTTP 服务 |

> `multi_agent_rag_service.py` 的 Web 服务端口为 **8008**；本版已合并修改版改动，见附录 A。

---

## 一、环境配置（`.env`）

`.env` 放在**本文件同级目录**（即 `F:\论文代码\`），程序启动时用 `load_dotenv()` 读取。

> ⚠️ **不要把 `.env` 交出去**：里面有真实密钥（大模型 Key / MinerU Token / Cloudflare Token / R2 密钥）。
> 本目录已加 `.gitignore` 排除 `.env`、`chroma_db/`、`raw_data/`、`2D_data/`、`__pycache__/` 等运行时产物；
> 交付/上传前请把 `.env.example` 复制成 `.env` 并填占位符。若曾经外发过本目录，请到各家后台**轮换全部密钥**。

### 1.1 核心变量（决定用哪家大模型）

| 变量 | 必填 | 说明 | 示例（占位符） |
|---|---|---|---|
| `OPENAI_API_KEY` | 是 | 大模型密钥。Cloudflare 以 `cfut_` 开头，DeepSeek / SiliconFlow 以 `sk-` 开头 | `sk-xxxxxxxx` |
| `OPENAI_BASE_URL` | 是 | OpenAI 兼容端点地址 | `https://api.deepseek.com` |
| `OPENAI_MODEL` | 是 | 模型名 | `deepseek-v4-flash` |
| `USE_RERANK` | 否 | `true` 开启 CrossEncoder 精排（更准但更慢），默认关 | `true` |

### 1.2 已内置的三套模型（改注释即可切换）

`.env` 里同时写了三家，同一时间只让一组生效（把其余组注释掉）：

| 供应商 | BASE_URL | 模型 | 备注 |
|---|---|---|---|
| Cloudflare Workers AI | `https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/v1` | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | 每日额度有限，容易 429 |
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` | 当前默认生效 |
| SiliconFlow | `https://api.siliconflow.cn/v1` | `tencent/Hunyuan-MT-7B` | 备用 |

切换方法：注释掉当前生效的那三行（`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`），取消注释目标供应商的三行，重启程序。

### 1.3 Cloudflare R2 对象存储（可选）

| 变量 | 说明 |
|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | 账户 ID，与 Workers AI 的 base_url 里那段一致 |
| `CLOUDFLARE_API_TOKEN` | 账户级 API Token（`cfat_` 开头，用于调用 R2 API） |
| `R2_ACCESS_KEY_ID` | R2 的 S3 兼容访问密钥 ID |
| `R2_SECRET_ACCESS_KEY` | R2 的 S3 兼容密钥 |
| `R2_ENDPOINT` | `https://<account_id>.r2.cloudflarestorage.com` |
| `R2_BUCKET` | 桶名（需在 R2 后台先创建，当前为空） |



## 二、HTTP 接口（`multi_agent_rag_service.py serve` 模式）

启动：`python multi_agent_rag_service.py serve`
默认地址：`http://127.0.0.1:8008`（启动后会自动打开浏览器）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回内置 Web 演示页（HTML，可直接提问 / 看画像 / 看决策链 / 跑评测 / 上传知识文件） |
| POST | `/ask` | 提问，走完整多 Agent 流程并返回答案（一次性返回；请求体可带 `"mode": "fast"|"full"`） |
| GET | `/ask_stream` | 提问，**SSE 流式**返回：节点进度 + 回答逐字推送（Web 页面默认走这个接口，query 参数 `mode=fast|full`） |
| GET | `/profile` | 查看长期记忆里的全部用户画像 |
| GET | `/audit` | 查看最近 20 条可追溯决策链 |
| POST | `/eval` | 跑 LLM-as-judge 评测，返回逐题得分与均值 |
| POST | `/ingest` | 上传多格式知识文件（PDF/Word/PPT/Excel/图片/txt）解析入库 |

### 2.1 `POST /ask` —— 提问

请求：
```json
{ "question": "突发口角歪斜肢体无力是怎么回事" }
```

响应：
| 字段 | 类型 | 说明 |
|---|---|---|
| `answer` | string | 最终答案正文 |
| `profile_hits` | list[string] | 本轮召回的用户画像（长期记忆命中的条目） |
| `event_log` | list[string] | 经过的节点名列表，如 `["router","retriever","analyzer","human_check","synthesizer"]` |
| `query_type` | string | 路由判定类型：`医疗健康` / `个人资料` / `学习资料` / `通用常识` / `量化决策` |

```json
{
  "answer": "……",
  "profile_hits": ["用户关注心脑血管急症识别"],
  "event_log": ["router", "retriever", "analyzer", "human_check", "synthesizer"],
  "query_type": "医疗"
}
```

说明：`/ask` 通过 state 字段 `skip_human_check` 跳过人工确认，保证 HTTP 接口不阻塞。
（本版改动：不再"保存全局变量 → 改 False → finally 恢复"，那种写法在并发请求下会互相覆盖恢复值，导致人工确认被静默永久关闭。）

### 2.1.1 `GET /ask_stream` —— SSE 流式提问

Web 页面默认走这个接口。事件为 `data: {json}` 帧，`type` 取值：

| type | 含义 |
|---|---|
| `start` | 开始（前端可清空回答框，带 `mode`） |
| `profile` | 长期记忆召回结果（`hits` 数组） |
| `node` | 某个节点执行完成（`name`：retriever / bridge_retrieve / fast_synth / router / analyzer / human_check / synthesizer / not_supported），前端据此画流程条 |
| `retrieval` | **召回证据明细**：`items` 按顺序列出检索到的资料块（`rank`/`sub`（来自第几个子查询）/`source`/`snippet`），`meta` 带 `rerank`（ok/skipped/degraded_timeout）、`top_sim`（贴合度）、`ms`（检索耗时）、`subs`（子查询列表）、`per_sub`/`groups`（分组配额）。用于把"精排/查询分解改变了什么"可视化 |
| `plan` | 多意图问题被拆解时推送，`subs` 为子查询列表 |
| `timing` | 节点耗时（`node`、`ms`），前端展示便于定位瓶颈 |
| `token` | 回答的文本增量（`text`），逐字上屏（流式阶段即渲染 Markdown 格式）；不调 LLM 的节点（如 `not_supported`）答案会在图执行完后按 40 字一段补推 |
| `done` | 结束，带 `mode`、`query_type`、`event_log`、`sources`、`ttft_ms`（首字耗时）、`total_ms`（总耗时）、`cached` |
| `error` | 异常（`message`），前端展示错误而不是静默断流 |

实现要点：节点内用 `Runner.run_streamed` 逐 token 读取，经 LangGraph 的 `custom` 流
（`get_stream_writer()`）透传到 SSE；`/ask_stream` 与 `/ask` 一样临时关闭人工确认。

### 2.1.3 能力边界与多跳兜底（三个节点）

| 机制 | 触发方式 | 行为 |
|---|---|---|
| **能力边界声明** | 本地关键词启发式（成功率/概率/排名/打分/哪家最好…）或 router 判定 `量化决策` | 走 `not_supported` 节点，**不调 LLM**，直接返回固定话术：说明暂不支持量化决策类问题，并给出可替代问法。返回 < 50ms |
| **多轮会话记忆（轻量槽位策略）** | 请求带 `session_id`（前端用 sessionStorage 生成并保持） | 会话态只存**结构化字段**：`slots.subject/domain` + `last_topic` + 最近 3 轮主题（**不存答案原文**，省 token 且避免串题）。上限 `SESSION_MAX`(50) / 过期 `SESSION_TTL`(2h) 自动清理 |
| **槽位填充（本地规则，0 LLM）** | 本轮被判定为"追问"（含 他/她/它/这个/上面… 或短句省略主语）且会话里有 `subject` | 把追问补成可检索的完整问题：`需要拍片吗` → **`胸痛 需要拍片吗`**（用 `_fill_slots`，毫秒级字符串拼接）。**填充结果只用于检索，生成阶段仍用用户原话** |
| **澄清反问（能力边界之一）** | 指代型追问但会话无线索（首轮就说"他"） | 走 `need_clarify` 节点**直接反问**，不检索不调 LLM（实测 **4ms**）；避免罗列全部候选 |
| **LLM 改写兜底** | 追问且本地填充失败但会话有线索 | `query_planner` 增加 `resolved_query` 字段（含 few-shot），把指代替换成上文实体后再检索 |
| **追问答案不入全局缓存** | 追问轮自动绕过答案缓存（读与写都跳过） | 追问答案依赖会话上下文，避免其他会话读到别人的上下文答案 |

关键实现点：`_extract_slots()` 回写槽位时**优先取命中资料的标题**（排除"目录/索引/使用说明"样板段，且对"胸痛的就诊科室与病情分级"这类标题取"的"前主体 → `胸痛`）；追问轮**不覆盖**已有 `subject`；澄清/边界轮**不回写**槽位。
| **多跳实体桥接（条件化）** | 含强关系词（创始人/毕业于/母公司…）**且**首跳 0 命中或证据强度低于 `BRIDGE_MIN_CONF`（默认 0.55）时触发 | `bridge_retrieve` 节点：先让 bridge Agent 定位中间实体（如"DeepSeek 创始人"→梁文锋，并标注依据是知识库还是模型常识），再用"实体 + 原问题"做第二跳检索，合并去重后交给生成节点。链路：`retriever → bridge_retrieve → fast_synth/analyzer` |
| **来源分级标注** | 所有生成节点（共享系统提示词第 7 条） | 每条结论必须标注「【知识库·文件名】」或「【模型常识·未经核实】」；知识库无资料时开头写明「知识库未收录」，禁止把模型常识伪装成资料依据，也不得在无支撑时给出精确数字 |

设计意图：面对知识库覆盖不到或超出能力边界的问题，系统选择**声明边界 + 标注来源 + 提示缺口**，而不是生成看似可信的结论。

### 2.1.4 性能设计（三处优化）

| 优化 | 说明 | 效果 |
|---|---|---|
| **链路开关** | Web 问答可选两条链路：`mode=fast`（默认，`START → retriever → fast_synth → END`，1 次 LLM 调用）与 `mode=full`（多 Agent 协作：`router → retriever → analyzer → human_check → synthesizer`，3 次 LLM 调用 + 人工确认路径）。页面输入框右侧下拉框即可切换；两条链路的答案缓存相互隔离（命名空间 `fast` / `full`） | 快速：约 4.3s；完整：约 10s+，用于演示协作与 HITL |
| **精排切片顺序（重要修复）** | 候选池 `RERANK_POOL`(默认 6，已校对) 先交精排，**精排后**再切最终 `top_k`。原实现"先切 top_k 再精排"，导致 RRF 排名 4~8 的正确段落永远进不了精排视野（实测：多意图查询下正确段 RRF 排 #5，被切掉后精排无法救回） | 修复后多意图场景正确段从"召回不到"变为精排后 #2 |
| **查询规划（query_planner：分解 + 多跳判定合并）** | 多意图启发式 **或** 强关系词触发；两者都不满足则**完全不调用** | 检索前**一次** LLM 调用产出完整检索计划：`sub_queries`（多意图拆成 2~3 条，带 few-shot 约束）、`need_bridge`（是否跨实体多跳）、`bridge_entity`（中间实体候选）。原先由 planner + bridge_extractor 两个 Agent 分别判断会冲突，合并后消除叠加歧义并省一次往返 |
| **分组配额召回** | 按计划逐个子查询"独立多路召回 + 独立精排" → 各自保留 topN（`RETRIEVE_PER_SUB`，默认 6，已校对）→ **交错合并去重** → 总上限 `RETRIEVE_MAX_TOTAL`（默认 9，已校对） | 解决"一个向量表达两个意图互相稀释"与"先检索的子查询把名额占满"。实测双意图：胸痛正确段 **#1**、另一意图证据 **#2** |
| **多跳桥接（条件化 + 实体复用）** | 仅当规划 `need_bridge=true` **且**（首跳 0 命中 或 证据强度 `retrieval_conf` < `BRIDGE_MIN_CONF`）才触发 | 中间实体**优先复用规划给的候选**（省一次 LLM），规划没给才由 `bridge_extractor` 从资料/常识现场抽取；二跳用"实体＋原问题"检索并合并去重。实测纯多跳问题 LLM 调用从 2 次降到 **1 次** |
| **多子查询的耗时控制** | 多子查询时用更小的精排候选池 `MULTI_SUB_POOL`（默认 9，单查询用 `RERANK_POOL`；原文档写的 6 已校正为 9） | 实测双子查询检索约 6.5~7.0s（planner ~1s + 2 次精排）；调小 `RETRIEVE_PER_SUB=2`、`MULTI_SUB_POOL=4` 可压到约 4s |
| **精排动态门控** | `USE_RERANK=true` 时按规则决定是否精排：候选≤1 → 不跑；含指代词（它/这个/上面…）或强关系词（创始人/毕业于…）→ 强制跑；其余 → 跑。精排走线程池并设 `RERANK_TIMEOUT`（默认 2.0s，可用 `.env` 调大），超时降级用 RRF 顺序，状态写入审计日志（`rerank=ok/skipped/degraded_timeout`） | 实测 8 候选精排约 0.9~2.9s；关闭时检索约 45~55ms |
| **精排门控的实测结论** | 两条常见门控思路在本项目被数据证伪：① 用 RRF top1/top2 分差判歧义——RRF 分数粒度仅约 0.0003，实测 gap 多为 0.0，分不出来；② 用"top1 向量相似度高就跳过精排"——样板段落（目录/使用说明）相似度 0.757 **高于**真实答案段 0.72，向量相似度分不清"关键词密集的样板"与"真正回答的段落"。故 `RERANK_SKIP_CONF` 默认置空（不启用跳过） | 演示"精排价值"的可复现案例见下方语料说明 |
| **桥接条件化** | 只有「含强关系词」**且**「首跳 0 命中或证据强度 `retrieval_conf` < `BRIDGE_MIN_CONF`(0.55)」才做二跳。实测：相关段落 conf≈0.69（不桥接），无关/缺资料 conf≈0.41（桥接） | 避免无条件调用 LLM，省 ~2.8s；并修掉"这是谁的资料"类误触发 |
| **超时快速失败** | LLM 客户端 `LLM_TIMEOUT`(默认 30s，原 90s) + `LLM_MAX_RETRIES`(默认 0，原 1)：网关卡顿时快速失败让前端提示重试，而不是干等几十秒 | 最坏耗时从 180s 降到 30s |
| **快速通道** | `fast_mode=True` 时跳过 router 的 LLM 往返，分析+总结合并为一次流式调用；`fast_agent` 限 `max_tokens` | LLM 调用从 **3 次降到 1 次** |
| **答案缓存** | `AnswerCache` 两级：① 规范化问题精确匹配（落盘 `_answer_cache.json`）② 语义近似匹配（问题向量余弦 ≥ `CACHE_SEMANTIC_THRESHOLD`，默认 0.85）。实测同一问题不同说法相似度 0.873~0.950，不同问题仅 0.18~0.38，故 0.85 安全 | 命中后 **总耗时约 0ms** |
| **长度与超时控制** | 送 LLM 的资料块按 `DOC_CHAR_LIMIT`(500 字) 截断；答案 `ANSWER_MAX_TOKENS`(800)；LLM 客户端 `LLM_TIMEOUT`(90s) + 重试 1 次 | 降 prefill/生成时长；网关偶发卡顿不再拖死整轮 |

相关开关均可写在 `.env`：`CACHE_SEMANTIC_THRESHOLD`、`LLM_TIMEOUT`、`USE_RERANK`（CrossEncoder 精排，开启会加载 bge-reranker-v2-m3 并显著增加启动与每问耗时，演示环境建议关）。

说明：交互模式（不带 `serve`）与 `POST /eval` 固定走完整三段式（router → analyzer → human_check → synthesizer），用于演示多 Agent 协作与 HITL。

### 2.2 `GET /profile` —— 用户画像

```json
{ "profiles": ["用户是糖尿病患者，关注饮食控制"] }
```

### 2.3 `GET /audit` —— 决策链

返回最近 20 条，每条含 `ts`（时间）、`node`（节点）、`in`（输入摘要）、`out`（输出摘要）：

```json
{ "events": [ { "ts": "2026-09-03T16:00:00", "node": "router", "in": "query=...", "out": "need=True,type=医疗" } ] }
```

### 2.4 `POST /eval` —— 跑评测

无需请求体。对内置测试集逐题跑完整流程，由 judge Agent 打三项 0-1 分（约 1-2 分钟）：

```json
{
  "rows": [ { "q": "chmod 755 是什么意思", "context_precision": 0.9, "faithfulness": 1.0, "answer_relevancy": 0.95 } ],
  "avg": { "context_precision": 0.88, "faithfulness": 0.92, "answer_relevancy": 0.9 }
}
```

指标含义：
- `context_precision` 检索资料与问题的相关程度
- `faithfulness` 答案是否严格基于检索资料（不编造）
- `answer_relevancy` 答案是否切题、有用

### 2.5 `POST /ingest` —— 多格式知识上传（multipart/form-data，agent 驱动）

来自毕设摄入子系统（原 `E:\xuexi\lunwen2`）的代码，**已整体合并进 `multi_agent_rag_service.py`（第 16–19 节），不再有独立 `ingest.py`**，规则与其 README 一致：
原始文件落 `raw_data/`、处理产物落 `2D_data/`、扩展名规则分类、同文件同库重复导入幂等替换。

**agent 驱动**：上传后由摄入 agent（主文件第 19 节 `build_ingest_agent`）自主调用
`ingest_document` 工具完成"MinerU 识别 → 写向量库 → 同步 BM25"，响应里的 `trace`
是 agent 的完整调用轨迹（AI 决策 / 工具返回 / AI 总结）。

| 表单字段 | 类型 | 说明 |
|---|---|---|
| `file` | file | 知识文件，支持 PDF / docx / pptx / xlsx / xls / txt / md / png / jpg |
| `target` | string | 目标集合：单库架构固定 `medical`（可省略） |

响应：
```json
{ "ok": true, "source": "xxx.pdf", "target": "medical", "chunks": 12, "status": "ingested",
  "trace": ["AI 调用工具 ingest_document 参数=...", "工具返回：...", "AI 总结：..."] }
```

说明：
- txt/md 本地解析秒回；其余格式（含图片）走 MinerU 云端高精度解析（约 1~5 分钟），需 `.env` 配 `MINERU_API_TOKEN` + `USE_MINERU_API=true`
- 入库即写对应业务集合并同步 BM25 索引，**立刻可提问**；服务重启后自动从 `2D_data/` 恢复索引
- Web 演示页底部已带上传卡片，选文件 → 选目标库 → 上传导入，页面会显示 agent 调用轨迹

---

## 三、交互模式命令

直接运行（不带 `serve`）进入命令行问答：`python multi_agent_rag_service.py`

| 命令 | 作用 |
|---|---|
| `/profile` 或 `画像` | 查看长期记忆中的用户画像 |
| `/audit` 或 `审计` | 查看最近 10 条决策链 |
| `/eval` 或 `评测` | 跑 LLM-as-judge 评测并打印三项均值 |
| `exit` / `quit` / `退出` | 结束程序 |

交互模式下**默认开启人工确认**：analyzer 产出草稿后会暂停，终端提示 `[y 直接发布 / 输入修改意见]`，输入修改意见会让 synthesizer 据此调整最终答案。

---

## 四、邮件 Agent（`email_hitl_agent.py`）

> ⚠️ 注：`email_hitl_agent.py` 当前**不在本目录**（原 README 描述的是另一项目里的文件），需从原项目复制后才能按本节运行。

运行：`python email_hitl_agent.py`

流程：`classify`（判断邮件是否需要处理）→ 需要则进 `email_agent` → 调用 `send_email` 时**被 HITL 中间件拦截** → 人工确认 `y` 才真正发信，`n` 驳回。

| 工具 | 说明 |
|---|---|
| `list_recent_emails(n=5)` | 列出收件箱最近 n 封（序号 / 发件人 / 主题 / 日期） |
| `read_email(email_id)` | 按序号读取邮件全文（主题 + 正文前 500 字） |
| `send_email(to, subject, body)` | 发信，**挂了人工确认拦截** |

分类关键词：`周报` / `会议` / `评审` / `紧急` / `付款`，命中则 `need_handle`，否则直接结束。

---

## 五、运行环境与依赖
- 主要依赖（`multi_agent_rag_service.py`）：`openai-agents`、`langgraph`、`langchain`、`langchain-core`、`langchain-openai`、`openai`、`chromadb`、`sentence-transformers`、`rank-bm25`、`jieba`、`fastapi`、`pydantic`、`uvicorn`、`python-dotenv`、`requests`
- 邮件 Agent 依赖（`email_hitl_agent.py`）：`imaplib` / `smtplib` / `email`（均为标准库）
- 说明：PDF / Word / PPT / Excel / 图片的解析**全部走 MinerU 云端 REST API**，因此不再需要 `python-docx`、`openpyxl`、`PyMuPDF` 等本地解析库；`langchain-core` 与 `pydantic` 虽未在 README 旧版列出，但代码中直接 import
- 嵌入模型：`BAAI/bge-small-zh-v1.5`，通过 `HF_ENDPOINT` 走 `hf-mirror.com` 国内镜像（代码里已设置在 import 之前，不要改动顺序）
- 知识库目录：`medical_kb/`（**唯一知识库目录**，原"医疗/考试"双库已合并），放入 `.txt` 即可被自动加载建索引；多格式文件（PDF/Word/图片等）走 Web 上传（由摄入 agent 调度 `ingest_document` 工具入库）
- **改动 txt 资料不再需要手动重建索引**：块 id 已改为**内容寻址**（`txt_` + 来源/文本的 md5），启动时对 `medical_kb/` 做一次增量对账——新增段落入库、改动段落"删旧 id + 加新 id"、删除的段落从库里清掉，旧版位置 id（`medical_0..N`）在首次启动时自动迁移。日志会打印 `[索引] medical_kb txt 对账：新增 X / 删除 Y / 现有 N 块`。因此 `_rebuild_index.py` 不再需要（该文件本来也不在本目录）。
- **演示"精排价值"的对照语料**：`medical_triage.txt` 内植入了 3 个样板干扰段（索引/使用说明/术语对照，关键词密集但不含答案）与 8 个主题段（含三套同名"分级"体系）。**可复现的 A/B（查询「胸痛挂哪个科」，务必勾选"不用缓存"）**：

  | 配置 | 精排状态 | 检索耗时 | 召回顺序（首条 → 末条） |
  |---|---|---|---|
  | `USE_RERANK=false` | skipped | **55ms** | 1.本手册使用说明　2.就医常见问题索引　3.**胸痛的就诊科室与病情分级**（答案被挤到第 3） |
  | `USE_RERANK=true` | ok | **1244ms** | 1.**胸痛的就诊科室与病情分级**　2.本手册使用说明　3.就医常见问题索引 |

  说明：最终答案文本两种配置差别不大——因为 LLM 会把 3 块资料都读完，正确答案在第 3 条也能用；**精排的价值体现在"把真正的答案提到第 1 位"**，页面的「召回证据」面板会把这一点直接显示出来。
  原文件备份在同目录 `_medical_triage_original.txt.bak`
- **演示时必须勾选页面上的「不用缓存」**（对应 `nocache=1`）：否则相同问题第二次提问会命中答案缓存、总耗时 0ms、根本不走检索，看不到任何差异
- 摄入目录：`raw_data/`（上传的原始文件）与 `2D_data/`（解析产物 json，重启后自动恢复 BM25 索引），不要手动删除
- 非 txt 格式摄入需在 `.env` 追加：`MINERU_API_TOKEN`（mineru.net 创建）与 `USE_MINERU_API=true`
- 知识库为单目录 `medical_kb/`：所有资料（含上传导入）统一入这一个库。


---

## 附录 A · 修改清单（已合并到 `multi_agent_rag_service.py` / `frontend/index.html`）

> 行号以修改版文件为准；本版已把 `_v2` 副本合并回来，以下改动全部生效。

### A1 Bug 修复（高）

| # | 位置（v2） | 问题 | 改法 |
|---|---|---|---|
| 1 | `human_check_node` / `RAGState` / `run_turn` / `/ask` / `/ask_stream` / `/eval` | 三处端点用"保存全局 `HUMAN_REVIEW` → 改 False → finally 恢复"，并发下恢复值互相覆盖，人工确认会被静默永久关闭 | 新增 state 字段 `skip_human_check`，HTTP 端点传入 True；全局 `HUMAN_REVIEW` 只保留给交互模式做默认开关 |
| 2 | 前端 `formatAnswer` | LLM 答案直接拼 HTML 塞 `innerHTML`，无转义。答案复述上传文档内容 → 文档里写 `<img onerror=...>` 即执行（存储型 XSS） | 新增 `escHtml()`，`formatAnswer` 先整体转义再做 Markdown 转换 |
| 3 | 前端 done 事件 chip | `label` 里含路由类型（LLM 输出）与 `sources`（上传文件名），同样未转义即 `insertAdjacentHTML` | 拼装后 `escHtml(label)` |
| 4 | L156（`_load_txt_chunks`） | `open(encoding="utf-8")` 无容错，放一个 gbk 的 txt 服务启动即崩（对比 `parse_txt` 有 `errors="ignore"`） | 加 `errors="ignore"` |
| 5 | `/ingest`（`save_path`） | `os.path.join(RAW_DIR, file.filename)`，文件名传 `../../x.pdf` 可写到 raw_data 之外 | `os.path.basename(file.filename)` |

### A2 Bug 修复（中 / 低）

| # | 位置 | 问题 → 改法 |
|---|---|---|
| 6 | `retriever_node` 检索循环 | 证据强度原取各子查询 **max**：强意图掩盖弱意图，多跳子查询永远等不来桥接 → 改 **min**（任一子查询证据不足即值得补证据） |
| 7 | `retriever_node` | `stats` 是共享 dict，多子查询时前面的诊断被覆盖，前端只看到最后一个子查询的 rerank 状态 → 改为每子查询独立 `st_sq`，再聚合成 `stats`（rerank 状态拼接去重） |
| 8 | 前端 `newChat` / `openConv` | 不关进行中的 SSE，切换后 `cur` 失效导致 token 事件抛错、本轮存不进历史 → 两个函数开头 `es.close()` |
| 9 | 交互模式 `main()` 画像抽取 | 缺 HTTP 侧的兜底过滤，且每轮都抽 → 抽出公共 `_filter_profile_points()` 两处共用，并按 `PROFILE_EVERY` 频率抽取 |
| 10 | `_is_followup` | `"此"` 做子串匹配，"此外/因此/如此"大量误判成追问 → 从指代判定表里移除 |
| 11 | `retrieval_info` 的 `sub` | `gi_of.get(key, 1) + 1` 与注释"查不到记 1"不符（实际记 2）→ 默认改为 0 |
| 12 | `_expand_same_source` 注释 | "只命中 1 块降为 1 跳"与代码 `r_eff=0` 矛盾 → 注释改为与代码一致 |
| 13 | `ingest_document` 工具 docstring | 写"target 只能是 medical 或 kyca"，kyca 已删除，agent 可能白试一轮 → 改为"单库架构下固定 medical" |
| 14 | 查询规划兜底值 | 兜的是 router 形状 `{"needs_retrieval":...}`，与 planner 消费方字段对不上 → 改为规划形状 `{"sub_queries":[rq],"need_bridge":False,...}` |
| 15 | `_prepare_session` | `retrieval_query == question` 与 strip 后的查询比较，原问题带首尾空格时澄清/改写兜底双双失效 → 与 `question.strip()` 比较 |
| 16 | `SHARED_SYSTEM_PREFIX` | 条目编号 1,2,3,4,6,7 缺 5 → 重排为 1~6 |
| 17 | `chat_meta_node` | 文案"最近几轮（含本轮）"，但本轮被 `NON_RETRIEVAL_TYPES` 排除、根本不会进列表 → 去掉"（含本轮）" |
| 18 | `_fill_slots` 人称替换 | 按元组顺序替换，"他"先于"他们"，"他们在哪"→「obj们在哪」→ 按长度降序替换 |
| 19 | `RERANK_TIMEOUT` 注释 | 注释写"默认 1.5s"，代码是 2.0s → 注释对齐 |

### A3 优化（不影响功能）

| # | 位置 | 说明 |
|---|---|---|
| 20 | 检索循环 | 删掉 `_retrieval_confidence` 的**重复编码 + 重复向量查询**，直接复用检索内部已算好的 `top_sim`（每子查询省一次 encode + 一次 chroma query）；该函数已无调用者，一并删除 |
| 21 | `eval_collect` | 喂 judge 的 `ctx` 未截断（含相邻扩展块）→ 与生成路径一致按 `DOC_CHAR_LIMIT` 截断 |
| 22 | `EventLogger.log` | `records` 内存无限增长 → 只保留最近 500 条（落盘 jsonl 不受影响） |
| 23 | `parse_doc_mineru_api` | 解压到 `2D_data` 顶层，多次导入同名图片互相覆盖、产物堆积 → 解压到 `mineru_<ts>/` 子目录，md 也只在本目录内 glob |
| 24 | `_cli_client` | 无 timeout / 重试配置（默认 600s），与主链路 30s 不一致 → 对齐 `LLM_TIMEOUT` / `LLM_MAX_RETRIES` |
| 25 | `_guess_query_type` | `"我" in text` → 个人资料，"我怎么用tar"会误标 → 收窄为 `"我的"` |
| 26 | `__main__` | 端口 8000 → **8008**，浏览器自动打开地址同步；前端改读 `frontend/index_v2.html` |

### A4 行为变化（验证时留意）

- `/eval`（含交互模式的 `评测` 命令）现在**不再逐题暂停等人工确认**（`skip_human_check=True`），7 题自动跑完。
- 交互模式的用户画像抽取从"每轮都抽"改为按 `PROFILE_EVERY`（默认 3）轮抽一次，与 HTTP 端一致。
- 多子查询时桥接判定更敏感（conf 取 min），若发现"本来不桥接的现在频繁桥接"，把证据强度口径改回 max 或按"仅对 need_bridge 为 true 的子查询"取值。
