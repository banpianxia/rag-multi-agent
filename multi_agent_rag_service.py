# ============================================================
# multi_agent_rag_service.py  —  个人知识库多智能体问答系统（单文件版）
# 说明：本文件已把原 ingest.py（多格式摄入模块）整体合并进来，运行只依赖本文件，
#       不再 import ingest；前端除外，页面仍在 frontend/index.html，运行时从磁盘读取。
#       本版已合并"修改版"的全部改动（5 个高危 bug + 14 个中低危 bug + 6 项优化，
#       完整清单见 README.md 附录 A）；Web 服务端口为 8008。
#
# 一句话职责：用户上传任意格式资料（PDF/Word/PPT/Excel/图片/txt）-> agent 调工具走 MinerU
#             云端识别 -> 写入 Chroma 向量库 + 同步 BM25 -> 混合检索问答（多 Agent + 三层记忆 + HITL + 评测）
#
# 运行方式：
#   交互问答：F:\python\.venv\Scripts\python.exe multi_agent_rag_service.py
#   Web 服务：F:\python\.venv\Scripts\python.exe multi_agent_rag_service.py serve  -> http://127.0.0.1:8008
#
# 目录约定（沿用摄入子系统 README 的"原始/产物分离"规则）：
#   raw_data/     上传的原始文件（不改动）
#   2D_data/      解析产物 json（重启恢复 BM25 索引靠它）+ MinerU 解析明细
#   chroma_db/    Chroma 向量库落盘
#   medical_kb/   健康医疗类 txt 资料
#   frontend/     Web 演示页（本文件不包含前端代码）
# ============================================================

# ---- 0. 国内 HF 镜像 + SSL 适配（必须在 import sentence_transformers 之前设置）----
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像替代 huggingface.co
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "true"                # 镜像也连不上时的兜底

# ---- 1. 标准库与第三方依赖 ----
import sys
import glob
import json
import re
import time
import uuid
import hashlib
import asyncio
import datetime
import platform
from typing import TypedDict, Annotated, NamedTuple

import requests                                                 # MinerU 云 API 的 HTTP 调用
import numpy as np                                              # 答案缓存的语义匹配用矩阵一次算完
import chromadb                                                 # 向量库（持久化）
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

from openai import AsyncOpenAI                                  # 给 openai-agents 显式绑客户端
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, ModelSettings
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool                           # 摄入工具用 langchain tool 装饰器

# ---- 2. 路径与 .env ----
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"))                    # 全文件只在这里读一次 .env

platform.machine = lambda: "AMD64"                              # 兼容部分模型包对架构的探测

CHROMA_DIR = os.path.join(_BASE_DIR, "chroma_db")
MEDICAL_DIR = os.path.join(_BASE_DIR, "medical_kb")             # 唯一知识库目录（原"医疗/考试"双库已合并）
RAW_DIR = os.path.join(_BASE_DIR, "raw_data")                   # 上传原件
PROCESSED_DIR = os.path.join(_BASE_DIR, "2D_data")              # 解析产物
LOG_PATH = os.path.join(_BASE_DIR, "_memory_event_log.jsonl")   # 决策链审计日志
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---- 3. MinerU 云 API 配置（图片/复杂 PDF 的高精度解析方案，非本地包）----
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")            # mineru.net 创建的 Bearer token
MINERU_BASE = "https://mineru.net/api/v4"                       # v4 接口前缀
USE_MINERU_API = os.getenv("USE_MINERU_API", "false").lower() == "true"   # 非 txt 是否走云端

# ---- 4. 嵌入模型（全进程唯一实例，摄入与检索共用）----
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# ---- 5. 向量库与两个业务集合 ----
client = chromadb.PersistentClient(path=CHROMA_DIR)
# 单库架构：原"医疗库(kb_medical)/考试库(kyca_kb)"已合并为一个知识库集合
try:
    client.delete_collection("kyca_kb")                          # 清掉历史遗留的空考试库集合
except Exception:
    pass                                                         # 不存在则忽略
kb_collection = client.get_or_create_collection(name="medical_kb", metadata={"hnsw:space": "cosine"})

FUSION_CANDIDATES = int(os.getenv("FUSION_CANDIDATES", "8"))    # 每路召回候选数（原 15，降到 8 减半精排计算量）
USE_RERANK = os.getenv("USE_RERANK", "false").lower() == "true" # 是否启用 CrossEncoder 精排（受 _should_rerank 门控）
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "2.0"))      # 精排超时阈值（秒）：超过则降级用 RRF 顺序
RERANK_POOL = int(os.getenv("RERANK_POOL", "6"))
# 精排候选池大小：先取这么多候选交精排，精排后再切最终 top_k。
# 设得比 top_k 大是必须的——否则 RRF 排名 4~8 的正确段永远进不了精排视野（原实现的缺陷）
# 阈值依据：实测 bge-reranker-v2-m3 对 8 个候选约 0.9s（冷启动 ~2.6s），
# 设 0.2s 会导致"每次都降级、精排永不生效"，故默认 2.0s（与上方 RERANK_TIMEOUT 一致）；想彻底不要精排就把 USE_RERANK 设为 false
if USE_RERANK:
    import concurrent.futures
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    # 精排队列：用线程池提交预测，主链路才能带超时等待（CPU 计算无法真正中断，
    # 超时只是"不再等它"，省的是响应延迟而不是算力）
    _RERANK_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank")
RRF_K = 60                                                      # RRF 平滑常数（业界常用 60）
RERANK_SKIP_CONF = os.getenv("RERANK_SKIP_CONF", "").strip()
# "高贴合度就跳过精排"的可选阈值：默认空=不启用。
# 为什么不默认开：实测样板段落（目录/使用说明）的向量相似度（0.757）反而高于真正答案段（0.72），
# 说明向量相似度无法区分"关键词密集的样板文字"和"真正回答问题的段落"，用它做门控会误判。
# 需要更激进地省算力时可在 .env 设 RERANK_SKIP_CONF=0.75 自行权衡

RETRIEVE_PER_SUB = int(os.getenv("RETRIEVE_PER_SUB", "6"))
# 查询分解后，每个子查询各自保留的条数（**分组配额**：各子查询互不挤占名额）
RETRIEVE_MAX_TOTAL = int(os.getenv("RETRIEVE_MAX_TOTAL", "9"))
# 多子查询合并后的总条数上限（= 子查询数 × PER_SUB，再封顶，避免上下文被撑爆）
MULTI_SUB_POOL = int(os.getenv("MULTI_SUB_POOL", "9"))
# 多子查询时每个子查询的精排候选池（比单查询小，控制"子查询数 × 单次精排"的总耗时）

NEIGHBOR_RADIUS = int(os.getenv("NEIGHBOR_RADIUS", "2"))
# 同源相邻块扩展半径：命中某块后，把同一文件里前后 r 块也带进上下文。
# 为什么需要：一份文档被切成多块后（一张信息图≈5 块），问整体问题时中间块常被
# 挤出 top_k（实测"学习路线分几步"漏掉第 5-6 周）——命中 1 块带出邻居即可补全
EXPAND_MAX_EXTRA = int(os.getenv("EXPAND_MAX_EXTRA", "10"))
# 相邻扩展最多额外带入的块数（控制上下文长度）

# 指代词/复杂结构词：出现时判定为复杂查询，强制走精排（与 _should_rerank 配合）
# 代词表（单一来源）：原来精排门控与追问判定各写一份、内容大量重复，改一处忘一处
# （"他"漏在表外就是这么踩出来的）；追问判定对"这个/那个"另有定语约束，故不复用本表，见 _is_followup
PRONOUN_KEYS = ("他", "她", "它", "这个", "那个", "上面", "刚才", "前者", "后者", "这位", "那位", "这些", "此")
ANAPHORA_KEYS = PRONOUN_KEYS + ("以及", "对比", "分别")            # 精排门控：指代或并列结构的"复杂查询"

def _should_rerank(query: str, n_cand: int, top_sim: float = None) -> bool:
    # 干什么：动态决定这次检索要不要跑精排（不是所有查询都值得花这笔计算）
    # 参数 top_sim：top1 向量相似度（1 - chroma distance），越大说明资料越贴合问题
    # 规则：① 总开关关或候选≤1 → 不跑；② 指代词/强关系词 → 强制跑；
    #      ③ 资料高度贴合（相似度≥RERANK_SKIP_CONF）→ 跳过；④ 其余（语义模糊/贴不上）→ 精排裁决
    if not USE_RERANK or n_cand <= 1:
        return False
    if any(k in query for k in ANAPHORA_KEYS):
        return True                                             # 指代型查询歧义大，必须精排
    if any(k in query for k in BRIDGE_KEYS):
        return True                                             # 多跳关系型查询，必须精排
    if RERANK_SKIP_CONF and top_sim is not None and top_sim >= float(RERANK_SKIP_CONF):
        return False                                            # 仅在显式配置阈值时才按"贴合度"跳过
    return True

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("缺少依赖 rank_bm25，请运行：F:\\python\\.venv\\Scripts\\python.exe -m pip install rank_bm25")
    raise

# ============================================================
# 6. txt 资料加载（medical_kb 单库，下划线开头的文件跳过）
# ============================================================
def _load_txt_chunks(directory: str) -> list:
    # 干什么：把目录下所有 .txt 按空行切段，产出 [{"text","source"}] 结构
    # 参数 directory：知识库目录绝对路径
    chunks = []
    for path in glob.glob(os.path.join(directory, "*.txt")):
        fname = os.path.basename(path)
        if fname.startswith("_"):
            continue                                            # 下划线开头视为说明文件，不建索引
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for p in paras:
            chunks.append({"text": p, "source": fname})
    return chunks

def load_medical_chunks():
    return _load_txt_chunks(MEDICAL_DIR)

# ============================================================
# 7. 分词 + 检索注册表构建（注册表 = 向量集合 + BM25 语料 + id 映射）
# ============================================================
def tokenize(text):
    # 干什么：中文分词；BM25 必须喂"已分词"文档，否则中文整句当一个词
    try:
        import jieba
        return [t for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return list(text)

TXT_ID_PREFIX = "txt_"          # 内容寻址块 id 前缀（txt 知识库）
LEGACY_TXT_PREFIX = "medical_"  # 旧版位置 id 前缀（medical_0..N），启动对账时清除，升级即自动迁移
ING_ID_PREFIX = "ing_"          # 摄入块 id 前缀（_file_key 生成）；txt 对账只统计、不动手

def _txt_chunk_ids(chunks, tag=TXT_ID_PREFIX):
    # 干什么：给 txt 段落生成**内容寻址**的块 id（来源 + 文本的 md5），文本不变 id 就稳定
    # 为什么：位置 id（medical_0..N）一改资料就整体错位（旧向量被解释成别的文本）；
    #         内容 id 让"改 / 删 / 加"资料都退化成一次增删，不必再手动重建整个索引
    # 去重：同一文件里出现完全相同的段落时用 _2/_3 后缀区分，避免同 id 互相覆盖
    seen, ids = {}, []
    for c in chunks:
        base = tag + hashlib.md5((c["source"] + "\x00" + c["text"]).encode("utf-8")).hexdigest()[:16]
        n = seen.get(base, 0)
        seen[base] = n + 1
        ids.append(base if n == 0 else f"{base}_{n + 1}")
    return ids

def _build_one(collection_obj, chunks, tag=TXT_ID_PREFIX):
    # 干什么：把磁盘上的 txt 语料"对账"进 chroma（内容寻址 id 增量增删），再建 BM25 与 id 映射
    # 参数 collection_obj：chroma 集合；chunks：[{"text","source"}]；tag：内容寻址 id 前缀
    # 为什么不再"集合非空就整体跳过"：旧版块 id 是位置 id（medical_0..N），集合一非空就永不更新
    #   ① 往 medical_kb/ 增删改 txt 完全不生效；② 内存语料与 chroma 内容一旦错位，
    #   旧向量命中的 id 会被 id2idx 解释成**另一段文本**（RRF 分数记到错文档上，静默出错不报异常）
    # 现在：id = md5(source + 文本)，文本不变 id 不变 -> 改了内容自动"删旧 id + 加新 id"，无需手动重建索引
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"]} for c in chunks]
    ids = _txt_chunk_ids(chunks, tag)
    old_ids = set(collection_obj.get(include=[]).get("ids") or [])
    # 只管辖 txt 前缀（含旧版 medical_ 位置 id）；摄入块 ing_* 不归这里管，绝不能误删
    managed = {i for i in old_ids if i.startswith(TXT_ID_PREFIX) or i.startswith(LEGACY_TXT_PREFIX)}
    stale = sorted(managed - set(ids))                          # 内容已改 / 资料已删：按旧 id 清除
    text_of = {cid: (t, m) for cid, t, m in zip(ids, docs, metas)}
    missing = [cid for cid in ids if cid not in old_ids]        # 新增资料 or 改内容后的新 id
    if stale:
        collection_obj.delete(ids=stale)
    if missing:
        collection_obj.add(ids=missing,
                           documents=[text_of[i][0] for i in missing],
                           embeddings=EMBED_MODEL.encode([text_of[i][0] for i in missing],
                                                         normalize_embeddings=True),
                           metadatas=[text_of[i][1] for i in missing])
    print(f"[索引] {collection_obj.name} txt 对账：新增 {len(missing)} / 删除 {len(stale)} / "
          f"现有 {collection_obj.count()} 块（内容寻址 id，改资料自动增量重建）")
    # 空库防护：用户还没放资料时 BM25Okapi 对空语料会抛异常，置 None 让检索只走向量路
    bm25 = BM25Okapi([tokenize(d) for d in docs]) if docs else None
    id2idx = {cid: i for i, cid in enumerate(ids)}
    print(f"[BM25] {collection_obj.name} 已建索引，语料 {len(docs)} 块")
    return {"collection": collection_obj, "bm25": bm25, "docs": docs, "metas": metas, "id2idx": id2idx}

# ============================================================
# 8. RRF 混合检索（向量路 + BM25 路 -> 倒数排名融合 -> 可选 CrossEncoder 精排）
# ============================================================
def _hybrid_retrieve(query, reg, top_k=5, stats=None, pool=None):
    # 参数 query：用户问题；reg：检索注册表；top_k：最终返回条数
    # 参数 stats：可选字典，回填本次检索的诊断信息（rerank 状态、候选数），供调用方记录
    # 参数 pool：精排候选池大小（缺省用 RERANK_POOL）；多子查询场景可传更小的值控制总耗时
    # 返回：[(文本, 元数据)] 列表
    coll, bm25, docs, metas, id2idx = reg["collection"], reg["bm25"], reg["docs"], reg["metas"], reg["id2idx"]
    q_text = "为这个句子生成表示以用于检索：" + query          # bge 系列建议的查询侧前缀
    q_emb = EMBED_MODEL.encode([q_text], normalize_embeddings=True)
    n_total = coll.count()
    if n_total == 0:
        return []                                              # 空库直接返回，不再走向量查询
    k_vec = min(FUSION_CANDIDATES, n_total)
    res = coll.query(query_embeddings=q_emb, n_results=k_vec, include=["documents", "metadatas", "distances"])
    # top1 相似度直接取自这次向量查询的 distance，不额外多算一次
    dists0 = res.get("distances") or []
    top_sim = round(1.0 - float(dists0[0][0]), 4) if dists0 and dists0[0] else 0.0
    if stats is not None:
        stats["top_sim"] = top_sim
    vec_ids = res["ids"][0]
    vec_rank = {vid: i + 1 for i, vid in enumerate(vec_ids)}
    fused_idx = {}                                             # RRF 融合表：语料下标 -> 累计得分
    # 向量路：chroma 命中的 id 经 id2idx 换算成语料下标后计 RRF 分
    for vid in vec_ids:
        idx = id2idx.get(vid)
        if idx is None:
            continue
        fused_idx[idx] = fused_idx.get(idx, 0) + 1.0 / (RRF_K + vec_rank[vid])
    # BM25 路：空库时 bm25 为 None，只走向量路
    if bm25 is not None and docs:
        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_order = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:FUSION_CANDIDATES]
        for pos, i in enumerate(bm25_order):
            fused_idx[i] = fused_idx.get(i, 0) + 1.0 / (RRF_K + (pos + 1))
    ordered = sorted(fused_idx.items(), key=lambda x: x[1], reverse=True)
    if stats is not None and ordered:
        # 回传 RRF 分数与 top1/top2 分差（诊断用；RRF 粒度约 0.0003，不适合做门控）
        stats["rrf_top1"] = round(ordered[0][1], 5)
        stats["rrf_top2"] = round(ordered[1][1], 5) if len(ordered) > 1 else 0.0
        stats["rrf_gap"] = round(stats["rrf_top1"] - stats["rrf_top2"], 5)
    # 【顺序关键】先取"精排候选池"（比最终 top_k 大），精排完再切 top_k。
    # 原实现先切 top_k 再精排，导致 RRF 排名 4~8 的正确段根本没机会被精排捞回来
    pool_n = max(top_k, pool or RERANK_POOL)
    cand_docs = [docs[i] for i, _ in ordered][:pool_n]
    cand_metas = [metas[i] for i, _ in ordered][:pool_n]
    if _should_rerank(query, len(cand_docs), top_sim):          # 动态门控：按"资料贴合度"决定是否精排
        pairs = [(query, d) for d in cand_docs]                 # 全部候选一次性组批（不是 for 逐个推理）
        try:
            future = _RERANK_POOL.submit(reranker.predict, pairs)
            scores = future.result(timeout=RERANK_TIMEOUT)      # 超时阈值内不返回就降级
            ranked = sorted(zip(cand_docs, cand_metas, scores), key=lambda x: x[2], reverse=True)
            cand_docs = [r[0] for r in ranked]                  # 精排只负责"排序"，不在这里切 top_k
            cand_metas = [r[1] for r in ranked]
            if stats is not None:
                stats["rerank"] = "ok"
        except concurrent.futures.TimeoutError:
            # 精排超时：直接用 RRF 融合顺序返回，保证请求链路不被拖住（线程仍在后台跑完）
            if stats is not None:
                stats["rerank"] = f"degraded_timeout>{RERANK_TIMEOUT}s"
        except Exception as e:
            if stats is not None:
                stats["rerank"] = f"error:{type(e).__name__}"    # 精排异常也不影响主流程
    else:
        if stats is not None:
            stats["rerank"] = "skipped"
    # 统一在这里切最终条数：候选池 → 精排 → top_k，保证精排能影响最终入选的那几条
    cand_docs = cand_docs[:top_k]
    cand_metas = cand_metas[:top_k]
    if stats is not None:
        stats["cand"] = len(cand_docs)
    return list(zip(cand_docs, cand_metas))

def retrieve(query, top_k=5, stats=None, pool=None):
    # 干什么：唯一知识库的混合检索入口（原"医疗/考试"分库已合并为单库）
    return _hybrid_retrieve(query, MED_KB, top_k, stats=stats, pool=pool)

def _expand_same_source(hits, reg, radius=None, max_extra=None):
    # 干什么：同源相邻块扩展（small-to-big 的简化版）——命中某块后，把同一文件里
    #        前后 NEIGHBOR_RADIUS 块也补进上下文（标记 expanded=True 便于前端识别）
    # 为什么：文档切块后语义连续，宽泛问题常只命中头尾块、中间块被挤出 top_k
    #        （实测：学习路线时间线.png 切 5 块，问整体漏掉第 5-6 周）
    # 参数 hits：[(文本, 元数据)]；reg：检索注册表（含全量 docs/metas，用于定位邻居）
    radius = NEIGHBOR_RADIUS if radius is None else radius
    max_extra = EXPAND_MAX_EXTRA if max_extra is None else max_extra
    if not hits or not reg:
        return hits
    docs, metas = reg["docs"], reg["metas"]
    # 建立"块指纹 -> 全库下标"的定位表（与去重键同规则：前 80 字 + 来源）
    pos = {}
    for i, (t, mt) in enumerate(zip(docs, metas)):
        pos.setdefault((t[:80], mt.get("source")), i)
    # 扩展优先级：同文件命中 ≥2 块的（说明整个文件相关）按全额半径扩；
    # 只命中 1 块的文件不扩展（r_eff=0）——单块命中的邻居大概率与本问题无关，
    # 还会把扩展预算耗光（实测踩过：医疗干扰块把预算吃完，真正需要的同文件邻居没被带上）
    from collections import Counter
    fam = Counter(m.get("source") for _, m in hits)
    out, seen = [], set()
    added = 0
    for t, m in hits:
        key = (t[:80], m.get("source"))
        i = pos.get(key)
        if key not in seen:                             # 已作为邻居扩展进来的命中块不重复追加，
            out.append((t, m))                          # 但仍要作为锚点继续扩展它的邻居（否则链中断）
            seen.add(key)
        if i is None:
            continue
        # 单块命中的文件不扩展：它的邻居大概率与本问题无关，还会把扩展预算耗光
        # （实测：医疗干扰块把预算吃完，导致真正需要的同文件邻居没被带上）
        r_eff = radius if fam[m.get("source")] >= 2 else 0
        for r in range(1, r_eff + 1):
            for j in (i - r, i + r):                    # 先近后远：i±1 优先于 i±2
                if j < 0 or j >= len(docs) or added >= max_extra:
                    continue
                if metas[j].get("source") != m.get("source"):
                    continue                            # 邻居不许跨文件（文件边界处 j±1 是别的文件）
                nkey = (docs[j][:80], metas[j].get("source"))
                if nkey in seen:
                    continue
                seen.add(nkey)
                nm = dict(metas[j])
                nm["expanded"] = "相邻扩展"              # 标记来源，前端/日志可区分
                out.append((docs[j], nm))
                added += 1
    return out

# ============================================================
# 9. 启动建库 + 恢复历史摄入索引
# ============================================================
MEDICAL_CHUNKS = load_medical_chunks()
MED_KB = _build_one(kb_collection, MEDICAL_CHUNKS, TXT_ID_PREFIX)

# 集合绑定表：摄入流程按目标名取集合与注册表（单库后仅 medical 一个目标，保留表结构兼容旧调用）
KB_BINDINGS = {"medical": (kb_collection, MED_KB)}

def register_extra(reg: dict, ids: list, chunks: list, metas: list):
    # 干什么：把摄入块追加进某个集合的 BM25 注册表（reg 结构见 _build_one）
    # 为什么必须做：融合时 vec 命中的 id 要查 reg["id2idx"] 换算成 BM25 语料下标，
    #              摄入块不在注册表里就会被匹配阶段丢弃
    start = len(reg["docs"])                                   # 追加前的语料长度 = 新块起始下标
    reg["docs"].extend(chunks)                                 # BM25 语料追加
    reg["metas"].extend(metas)                                 # 来源元数据同步追加
    reg["bm25"] = BM25Okapi([tokenize(d) for d in reg["docs"]])  # 全量重建（BM25Okapi 不支持增量）
    for i, cid in enumerate(ids):
        reg["id2idx"][cid] = start + i                         # chroma id -> 语料下标

def register_ingested_on_startup(reg_map: dict):
    # 干什么：启动时把 2D_data 里历史摄入记录重新注册进 BM25 注册表
    # 为什么：chroma 落盘持久、BM25 每次启动内存重建；只按 txt 重建会漏掉摄入块
    # 防重：同一批 ids 在同一集合只注册一次（历史遗留的同 id 多 json 不会造成重复语料）
    n_files = 0
    seen_id_sets = set()
    for jp in glob.glob(os.path.join(PROCESSED_DIR, "ing_*.json")):
        with open(jp, encoding="utf-8") as f:
            rec = json.load(f)
        target = rec.get("target", "medical")
        reg = reg_map.get(target)
        if reg is None:
            continue
        fingerprint = (target, tuple(rec.get("ids") or []))
        if fingerprint in seen_id_sets:
            continue
        seen_id_sets.add(fingerprint)
        register_extra(reg, rec["ids"], rec["chunks"],
                       rec.get("metadatas") or [{"source": rec["source"]}] * len(rec["chunks"]))
        n_files += 1
        print(f"[摄入恢复] {rec['source']} -> {target} 集合 {len(rec['chunks'])} 块已并入 BM25")
    if n_files:
        print(f"[摄入恢复] 共恢复 {n_files} 个历史摄入文件的索引")

register_ingested_on_startup({"medical": MED_KB})

def _check_ingest_orphans(reg: dict):
    # 干什么：只告警不删除——chroma 里有、但 BM25 注册表里没有的摄入块
    # 为什么会有：写 chroma 成功而 2D_data/*.json 落盘失败（或 json 被手工删掉）时，
    #             这些块在向量路会被 id2idx 查不到而静默跳过（_hybrid_retrieve 里 continue），
    #             等于"占了库容量但永远检索不到"。这里点名让人看见，交给人工/重导入处理
    known = set(reg["id2idx"])
    orphan = [i for i in (kb_collection.get(include=[]).get("ids") or [])
              if i.startswith(ING_ID_PREFIX) and i not in known]
    if orphan:
        print(f"[摄入对账] 警告：{len(orphan)} 个摄入块在 chroma 中但不在 BM25 注册表（2D_data 缺记录），"
              f"当前检索不到：{orphan[:3]}… 建议重新导入对应文件")

_check_ingest_orphans(MED_KB)

# ============================================================
# 10. 三层记忆：长期画像（Chroma）+ 决策链审计日志 + 短期对话（LangGraph checkpointer）
# ============================================================
PROFILE_MIN_SIM = float(os.getenv("PROFILE_MIN_SIM", "0.50"))
# 长期画像召回阈值：chroma cosine distance ≤ 此值才注入 prompt。
# 不设阈值的后果：打招呼（"你好"）也会拉回不相干画像（实测问题）。
# 实测校准（bge-small-zh）：无关寒暄 0.579；相关提问 0.233~0.485 → 取 0.50 分界，可按 PROFILE_MIN_SIM 调整

class ProfileMemory:
    def __init__(self, chroma_client, collection_name="user_profile"):
        self.coll = chroma_client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, points: list):
        # 干什么：写入长期画像，写入前做相似度去重（距离 < 0.15 视为已有）
        for p in points:
            if not p.strip():
                continue
            emb = EMBED_MODEL.encode([p], normalize_embeddings=True)
            res = self.coll.query(query_embeddings=emb, n_results=1)
            if res["distances"] and res["distances"][0] and res["distances"][0][0] < 0.15:
                print(f"[长期记忆] 已存在相似画像，跳过去重：{p}")
                continue
            pid = f"p_{uuid.uuid4().hex[:8]}"
            self.coll.add(ids=[pid], documents=[p], embeddings=emb, metadatas=[{"type": "profile"}])
            print(f"[长期记忆] 新增画像：{p}")

    def recall(self, query: str, top_k=6) -> list:
        # 干什么：按语义召回相关画像；空库返回空列表
        # 阈值：chroma cosine distance < PROFILE_MIN_SIM 才算相关——否则打招呼（"你好"）
        # 也会拉回不相干的画像（实测问题）。bge 归一化向量下，相关对通常 <0.45，无关 >0.6
        if self.coll.count() == 0:
            return []
        emb = EMBED_MODEL.encode([query], normalize_embeddings=True)
        res = self.coll.query(query_embeddings=emb, n_results=top_k, include=["documents", "distances"])
        hits = []
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            if float(dist) <= PROFILE_MIN_SIM:
                hits.append(doc)
        return hits

    def dump_all(self) -> list:
        if self.coll.count() == 0:
            return []
        return self.coll.get(include=["documents"])["documents"]

class EventLogger:
    # 干什么：把每个节点的输入/输出摘要落盘成 jsonl，供 /audit 追溯决策链
    def __init__(self, path):
        self.path = path
        self.records = []

    def log(self, node: str, in_summary: str, out_summary: str) -> dict:
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "node": node,
            "in": in_summary[:200],
            "out": out_summary[:200],
        }
        self.records.append(rec)
        if len(self.records) > 500:                             # 内存上限：只留最近 500 条，防止长跑服务无限增长（落盘 jsonl 不受影响）
            del self.records[:len(self.records) - 500]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

def _add_logs(existing, new):
    # 干什么：LangGraph state 里 event_log 字段的 reducer（list 追加语义）
    return (existing or []) + (new or [])

profile_memory = ProfileMemory(client, "user_profile")
event_logger = EventLogger(LOG_PATH)

# ---- 10.5 性能相关：事件推送 / 长度控制 / 缓存 ----
def _emit(evt: dict):
    # 干什么：往 LangGraph 的 custom 流推一个事件（SSE 前端用）
    # 为什么：节点内部要汇报"耗时/进度"，只有 custom 流能透传到 HTTP 层
    # 非流式调用（交互模式、/ask）下 get_stream_writer 会 no-op 或抛异常，都被吞掉不影响主流程
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        return
    try:
        writer(evt)
    except Exception:
        pass

DOC_CHAR_LIMIT = 500        # 送入 LLM 的单个资料块最大字符数（控制 prefill，越大越慢）
ANSWER_MAX_TOKENS = 800     # 最终回答 max_tokens（控制生成时长）
CACHE_SEMANTIC_THRESHOLD = float(os.getenv("CACHE_SEMANTIC_THRESHOLD", "0.85"))
# 语义缓存阈值：问题向量余弦相似度 ≥ 此值视为同问。
# 实测 bge-small-zh 相似度分布：同一问题的不同说法 0.873~0.950，不同问题仅 0.18~0.38，
# 中间空档很大，所以取 0.85 既能命中"换说法"又几乎不会误配；想更保守可设 0.90
CACHE_MAX_ENTRIES = 200
ANSWER_CACHE_PATH = os.path.join(_BASE_DIR, "_answer_cache.json")

def _truncate_docs(docs, limit=DOC_CHAR_LIMIT):
    # 干什么：把检索到的资料块截断后再送 LLM，降低输入长度
    return [(t[:limit], m) for t, m in docs]

def _guess_query_type(question: str, docs) -> str:
    # 干什么：不用 LLM 的本地启发式类型判定（快速通道用，只做元数据标注与缓存标签）
    # 为什么：router 那次 LLM 往返在演示里性价比低，关键词/来源名足够区分主题
    text = question + " " + " ".join(str(m.get("source", "")) for _, m in (docs or []))
    if any(k in text for k in ("医疗", "糖尿病", "血压", "血糖", "急诊", "症状", "用药", "科室", "卒中", "胸痛")):
        return "医疗健康"
    if any(k in text.lower() for k in ("chmod", "tar", "linux", "命令", "代码", "开发", "实习", "简历", "技能")):
        return "学习资料"
    if "我的" in text or "资料" in text:                        # "我"子串会误中"我怎么用tar"这类问法，收窄为"我的"
        return "个人资料"
    return "通用常识"

# ---- 10.6 能力边界：量化决策类问题（成功率/排名/打分）明确声明不支持 ----
QUANT_KEYS = ("成功率", "录取概率", "多大把握", "胜算", "概率", "排名第", "排第几",
              "打分", "评分", "得分", "哪家最好", "哪个最好", "哪家更", "选哪家", "推荐哪家")

def _is_quant_request(query: str) -> bool:
    # 干什么：本地启发式识别"量化决策类"问题（要求成功率/排名/打分等量化结论）
    # 为什么用规则不用 LLM：这类说法关键词清晰，规则零成本零延迟，也不会误伤普通提问
    return any(k in (query or "") for k in QUANT_KEYS)

NOT_SUPPORTED_ANSWER = (
    "该问题属于 **量化决策类**（需要成功率、排名或打分这类量化结论），当前版本暂不支持这一能力。\n\n"
    "原因：这类结论必须基于结构化的岗位/要求数据做加权计算才可复现、可审计；"
    "单纯靠文本检索生成一个数字属于编造，所以系统选择明确声明边界，而不是给出无法核实的结论。\n\n"
    "换成下面这些问法，我能给出有依据的回答：\n"
    "- 「我资料里命中了 XX 岗位要求的哪些项，还缺哪些」——逐项比对\n"
    "- 「XX 公司和 YY 公司的要求有什么差别」——基于资料做对比\n"
    "- 「XX 岗位通常要求哪些技能」——基于知识库资料作答"
)

# 说明：能力边界节点 not_supported_node 与多跳桥接节点 bridge_retrieve_node 的实现
# 放在第 13 节节点区（那里才定义了 RAGState 与各 Agent），本处只放判定所需的启发式与话术常量

# ---- 10.7 多跳兜底：实体桥接检索（先定位中间实体，再用它做第二跳检索）----
BRIDGE_KEYS = ("创始人", "创办", "CEO", "董事长", "总裁", "毕业于", "母校", "母公司", "子公司",
               "总部", "由谁", "哪年成立", "属于哪个公司", "发明专利")
# 说明：只保留"强关系词"。像"谁的""作者"这类弱关系词会被普通问句命中（如"这是谁的资料"），
# 造成误触发，故剔除——桥接本就要配合证据强度判断，宁可少触发也别白花一次 LLM
BRIDGE_MIN_CONF = float(os.getenv("BRIDGE_MIN_CONF", "0.55"))
# 桥接触发阈值：首跳 top1 相似度低于此值视为"证据不足"。相似度取自各子查询检索 stats 的
# top_sim 最小值（retriever_node 内聚合），
# 实测中文相关段落通常 0.55~0.75，不相关 <0.4，取 0.55 为起始值，可按日志里的实际值校准

def _needs_bridge(query: str) -> bool:
    # 干什么：粗判问题里是否含"跨实体关系词"（创始人/毕业于/母公司…）
    # 注意：这只是必要条件，真正的桥接决策在 _should_bridge 里结合证据强度做
    return any(k in (query or "") for k in BRIDGE_KEYS)

# ---- 10.8 多意图判定：决定是否走"查询分解"（拆子查询分别检索）----
DECOMPOSE_KEYS = ("顺便", "另外", "以及", "还有", "同时", "分别", "再帮我", "顺便帮", "并且", "另一个问题")
QUESTION_MARKS = ("什么", "怎么", "哪个", "哪家", "为什么", "多少", "怎么办", "如何", "是不是")

def _needs_decompose(query: str) -> bool:
    # 干什么：本地启发式判断是否为"多意图问题"（需要拆成多个子查询分别检索）
    # 为什么用规则：拼接词（顺便/另外/以及…）非常明确，规则判定零成本；
    #              另外"长问题含 ≥2 个疑问点"也算多意图（如"胸痛挂哪个科，另外头晕怎么办"）
    q = query or ""
    if any(k in q for k in DECOMPOSE_KEYS):
        return True
    marks = sum(1 for k in QUESTION_MARKS if k in q)
    return len(q) >= 25 and marks >= 2

# ============================================================
# 10.9 多轮会话态（轻量策略）：结构化槽位 + 上一轮主题，不存答案原文
#   设计取舍：只存"槽位与主题"（几十字节），不存完整对话摘要——省 token、避免
#   把上文答案当依据（串题）；语义跨度长的摘要留作后续工作
# ============================================================
SESSION_STATE = {}                                              # session_id -> 会话态
SESSION_MAX = int(os.getenv("SESSION_MAX", "50"))               # 最多保留会话数（超出按最旧淘汰）
SESSION_TTL = float(os.getenv("SESSION_TTL", "7200"))           # 会话过期秒数（默认 2 小时）
OMIT_TAILS = ("吗", "呢", "吧", "怎么", "怎么办", "需要", "要不要", "可以", "行不行")

def _is_followup(query: str) -> bool:
    # 干什么：判断本轮是否"依赖上文的追问"（含指代词，或短句且省略了主体）
    # 为什么需要：这类问题直接检索必然跑偏（"他"没有任何语义），必须先做槽位填充或澄清
    # 注意：「这个/那个/这些」常作定语（如"我这个技术"），不能一律当指代——只有出现在句首或
    #      句子很短时才算真指代，否则会把"我这个技术进哪家企业成功率大"误判成追问
    q = (query or "").strip()
    if not q:
        return False
    if any(k in q for k in ("他", "她", "它", "上面", "刚才", "前者", "后者", "这位", "那位")):
        return True
    # 注意："此"不放进指代判定——"此外/因此/如此"是高频连词，子串匹配会把普通问句误判成追问（实测问题）
    if any(k in q for k in ("这个", "那个", "这些")) and (q.startswith(("这个", "那个", "这些")) or len(q) <= 12):
        return True
    # 短句 + 无独立疑问主体（不含"什么/哪个/怎么/为什么/多少/如何"）+ 疑问收尾 → 视为省略型追问
    has_own_subject = any(k in q for k in ("什么", "哪个", "哪家", "怎么", "为什么", "多少", "如何", "是什么"))
    return len(q) <= 12 and (not has_own_subject) and q.endswith(OMIT_TAILS)

def _gc_sessions():
    # 干什么：清理过期与超额会话，避免进程内存无限增长
    now = time.time()
    for k in [k for k, v in SESSION_STATE.items() if now - v.get("ts", 0) > SESSION_TTL]:
        SESSION_STATE.pop(k, None)
    if len(SESSION_STATE) > SESSION_MAX:
        olds = sorted(SESSION_STATE.items(), key=lambda kv: kv[1].get("ts", 0))
        for k, _ in olds[:len(SESSION_STATE) - SESSION_MAX]:
            SESSION_STATE.pop(k, None)

def _load_session(sid: str) -> dict:
    # 干什么：取会话态；不存在则初始化（并顺带做过期清理）
    if not sid:
        return {"slots": {}, "last_topic": {}, "recent": [], "ts": time.time()}
    _gc_sessions()
    st = SESSION_STATE.get(sid)
    if not st:
        st = {"slots": {}, "last_topic": {}, "recent": [], "ts": time.time()}
        SESSION_STATE[sid] = st
    st["ts"] = time.time()
    return st

def _save_session(sid: str, st: dict):
    # 干什么：回写会话态（含最后活跃时间，供 TTL 淘汰）
    if not sid or not isinstance(st, dict):
        return
    st["ts"] = time.time()
    SESSION_STATE[sid] = st

def _has_context(session: dict) -> bool:
    # 干什么：判断会话里有没有可用于消解指代的线索（槽位 subject 或上一轮主题）
    s = (session or {}).get("slots") or {}
    t = (session or {}).get("last_topic") or {}
    return bool((s.get("subject") or "").strip() or (t.get("query") or "").strip())

# 会话元问题（问"聊了什么"这类关于对话自身的问题）：直接答会话摘要，不检索不调 LLM
CHAT_META_KEYS = ("我们聊了什么", "聊了什么", "聊过什么", "刚才问了什么", "之前问了什么", "我问过什么",
                  "历史对话", "聊天记录", "对话记录", "上一轮问了什么", "上面聊的")

def _is_chat_meta(query: str) -> bool:
    # 干什么：识别"关于会话自身"的元问题（如"我们聊了什么"）
    # 为什么单列：这类问题检索知识库必然答不上（答案在会话里），必须走会话摘要节点
    return any(k in (query or "") for k in CHAT_META_KEYS)

# 人称代词（他/她）要求会话里有"人"类上下文，否则应澄清而不是硬填
PRONOUN_PERSON = ("他", "她", "他们", "她们", "这位", "那位", "此人", "该人", "本人")
PERSON_HINTS = ("创始人", "创办", "作者", "CEO", "董事长", "总裁", "老师", "同学", "经理",
                "医生", "博士", "教授", "是谁", "毕业于", "人名", "客户", "用户")
NON_PERSON_SUBJECT = ("病", "痛", "症", "分级", "制度", "标记", "索引", "说明", "指标", "系统", "命令")

def _needs_person_context(query: str) -> bool:
    # 干什么：判断本轮是否用了"人称代词"（他/她）——这类指代必须有"人"类对象才可填
    return any(k in (query or "") for k in PRONOUN_PERSON)

def _has_person_context(session: dict) -> bool:
    # 干什么：会话里是否存在"人"类线索（上一轮问题带创始人/作者等词，或槽位对象不是疾病/制度类）
    t = (session or {}).get("last_topic") or {}
    txt = str(t.get("query") or "")
    if any(k in txt for k in PERSON_HINTS):
        return True
    subj = ((session or {}).get("slots") or {}).get("subject") or ""
    return bool(subj) and not any(k in subj for k in NON_PERSON_SUBJECT)

def _person_object(prev_query: str) -> str:
    # 干什么：从上一轮问题里提取"人"的指称短语（如「asd公司的创始人是谁」→「asd公司的创始人」）
    # 为什么要这层：人称代词"他"要替换成一个**能检索的表述**，而"人"的表述通常带关系词
    q = prev_query or ""
    m = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{1,12}的(?:创始人|创办人|作者|CEO|董事长|总裁|老师|医生|经理|负责人|本人))", q)
    if m:
        return m.group(1)
    # 次级：问句本身就是"X是谁/叫什么"时，取"是谁"前的对象
    m = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})是(?:谁|什么人|哪位)", q)
    if m:
        return m.group(1)
    return ""

def _fill_slots(query: str, session: dict):
    # 干什么：**本地规则**做指代消解/槽位填充——把追问补全成可检索的问题（不调 LLM，毫秒级）
    # 参数 query：本轮原始问题；session：会话态（含 slots.subject 与 last_topic）
    # 返回：(检索用查询, 指代说明)；无法解析时原样返回原问题与空说明
    # 关键点：① 检索用**解析后**的查询；② 指代说明要一并交给生成阶段（否则模型仍不知道"他"是谁）
    q = (query or "").strip()
    s = (session or {}).get("slots") or {}
    subject = (s.get("subject") or "").strip()
    prev_q = ((session or {}).get("last_topic") or {}).get("query") or ""
    if not _is_followup(q):
        return q, ""
    # ---- A) 人称代词（他/她）：必须有"人"类上下文，且要替换成"人"的指称短语 ----
    if _needs_person_context(q):
        if not _has_person_context(session):
            return q, ""                                        # 无人物线索 → 交给澄清节点
        obj = _person_object(prev_q) or subject
        if obj:
            pron = next((p for p in PRONOUN_PERSON if p in q), "他")
            filled = q
            # 按"长词优先"替换：先换"他们/她们"再换单字"他/她"——
            # 否则"他们在哪"会先被"他"命中，替换成「obj们在哪」
            for p in sorted(PRONOUN_PERSON, key=len, reverse=True):
                filled = filled.replace(p, obj)                 # 「他是哪个学校毕业的」→「asd公司的创始人是哪个学校毕业的」
            return filled, f"问题里的「{pron}」指上文提到的『{obj}』"
        return q, ""
    # ---- B) 省略型追问：补上主体（槽位优先，其次上一轮问题）----
    if subject:
        return f"{subject} {q}", f"问题省略了主体，按上文讨论对象『{subject}』理解"
    if prev_q:
        return f"{prev_q} {q}", f"问题省略了主体，按上一轮问题『{prev_q}』理解"
    return q, ""                                                # 填不出来 → 交给澄清或 LLM 改写兜底

SLOT_HINT_KEYS = ("病", "痛", "症", "公司", "大学", "学校", "专业", "系统", "命令", "药", "指标", "分级")
BOILERPLATE_TITLES = ("目录", "索引", "使用说明", "术语", "免责", "说明")   # 样板段标题不能当讨论对象
SUBJECT_SPLIT = ("挂", "是", "有", "怎么", "如何", "需要", "要不要", "能不能", "该", "的",
                 "，", ",", "?", "？", " ", "；", ";")

def _lead_subject(query: str) -> str:
    # 干什么：从问句开头切出"讨论对象"候选（如「胸痛挂哪个科」→「胸痛」、「asd公司创始人的代步工具」→「asd公司创始人」）
    # 为什么这么做：问句里主体通常在最前面，遇到"挂/是/的/怎么"等词就是主体结束点
    q = (query or "").strip()
    parts = re.split("|".join(re.escape(m) for m in SUBJECT_SPLIT), q, maxsplit=1)
    cand = parts[0].strip() if parts else ""
    return cand if 2 <= len(cand) <= 12 else ""

def _extract_slots(query: str, docs: list, query_type: str, session: dict) -> dict:
    # 干什么：本轮结束后回写槽位——subject（讨论对象）、domain（主题）、last_topic（本轮问题+命中来源）
    # 为什么用启发式：槽位只需"够下一轮消解指代"，纯规则零成本；抽错也只是回退到澄清反问
    session = session if isinstance(session, dict) else {}
    slots = dict(session.get("slots") or {})
    followup = _is_followup(query)
    subj = ""
    if followup and (slots.get("subject") or "").strip():
        # ① 追问轮**优先保留原讨论对象**，不被本轮命中资料的标题覆盖
        #   （否则问"他是哪个学校毕业的"命中"分级护理制度"后会污染槽位，下一轮又跑偏）
        subj = (slots.get("subject") or "").strip()
    else:
        # ② 先试"问句开头的主体"（更贴合用户实际在问什么，如「asd公司的创始人是谁」→ asd公司）
        cand = _lead_subject(query)
        if cand and any(k in cand for k in SLOT_HINT_KEYS):
            subj = cand
        # ③ 再退到"本轮命中资料的标题"（问句里没有明确主体时，如"怎么用tar压缩文件"）
        if not subj:
            for t, _m in (docs or [])[:3]:
                m = re.match(r"^【([^】]{2,20})】", str(t).strip())
                if m and not any(b in m.group(1) for b in BOILERPLATE_TITLES):
                    title = m.group(1)
                    # 标题常是"胸痛的就诊科室与病情分级"这种形式，取"的"前的主体更利于槽位填充
                    head = title.split("的")[0].strip()
                    subj = head if 2 <= len(head) <= 10 else title[:12]
                    break
    if subj:
        slots["subject"] = subj
    if query_type:
        slots["domain"] = query_type
    session["slots"] = slots
    session["last_topic"] = {"query": query,
                             "sources": [m.get("source") for _t, m in (docs or [])][:3]}
    session["recent"] = ([{"q": query, "subject": slots.get("subject", "")}] +
                         (session.get("recent") or []))[:3]
    return session

def _session_hint(session: dict) -> str:
    # 干什么：把会话态压成一小段文字喂给规划 Agent（供 LLM 改写兜底用），控制在百字内
    s = (session or {}).get("slots") or {}
    t = (session or {}).get("last_topic") or {}
    parts = []
    if s.get("subject"):
        parts.append(f"上文讨论对象：{s['subject']}")
    if s.get("domain"):
        parts.append(f"主题领域：{s['domain']}")
    if (t.get("query") or ""):
        parts.append(f"上一轮问题：{t['query']}")
    return "；".join(parts)

NON_RETRIEVAL_TYPES = ("需要澄清", "量化决策", "会话回顾")   # 这些轮次不是检索结果，不该回写槽位
PROFILE_EVERY = int(os.getenv("PROFILE_EVERY", "3"))
# 长期画像抽取频率：每 PROFILE_EVERY 个"正常检索回答"轮，后台异步抽取一次画像入库。
# 为什么不是每轮：抽取要多花一次 LLM（约 1~2s），设为后台任务不阻塞响应但烧 token；
# 设 0 可完全禁用。画像只存"跨会话稳定的偏好/背景"，由抽取 Agent 自行判断

# 长期画像的后台任务集合：持有引用防止被垃圾回收（asyncio 只保留弱引用）
_PROFILE_TASKS = set()

def _filter_profile_points(raw_lines: list) -> list:
    # 干什么：画像候选行的统一清洗——去列表符号、滤空行/超长短句，再兜底过滤模型无视指令输出的解释文字
    # 为什么抽成公共函数：HTTP 后台抽取与交互模式两处都要同一段过滤；原版只有 HTTP 侧有过滤，
    # 交互模式直接 add，模型输出"未透露/不抽取…"这类解释文字会被当成画像入库，污染长期记忆
    points = [ln.strip().lstrip("-•*").strip() for ln in raw_lines if ln.strip()]
    points = [p for p in points if 6 <= len(p) <= 80]           # 过滤空行与超长短句
    return [p for p in points
            if not any(b in p for b in ("不抽取", "无法抽取", "未透露", "没有可抽取",
                                        "无可抽取", "不构成", "暂无", "不属于画像", "不输出"))]

def _schedule_profile_extraction(question: str, answer: str, session: dict) -> bool:
    # 干什么：判断本轮是否该抽长期画像；该抽就用后台任务跑（fire-and-forget，不阻塞响应）
    # 参数 question/answer：本轮问答原文（答案截断 1500 字，控制抽取成本）
    # 返回：是否真正调度了抽取（供前端提示"本轮已触发画像抽取"）
    session["turns"] = int(session.get("turns") or 0) + 1
    if PROFILE_EVERY <= 0 or session["turns"] % PROFILE_EVERY != 0:
        return False
    q_text, a_text = question, (answer or "")[:1500]

    async def _job():
        # 抽取失败只打日志，绝不影响主流程（画像本来就是锦上添花）
        try:
            res = await Runner.run(profile_extractor_agent,
                                   f"用户问题：{q_text}\n回答：{a_text}")
            points = _filter_profile_points(str(res.final_output).split("\n"))
            if points:
                # to_thread：add 内部要 encode + 查 chroma（同步 CPU/IO），别在事件循环里做
                await asyncio.to_thread(profile_memory.add, points)   # 内部自带相似度去重（<0.15 跳过）
                event_logger.log("profile_extractor", f"问题={q_text[:60]}",
                                 f"新增画像 {len(points)} 条：{points}")
                print(f"[长期记忆] 后台新增画像 {len(points)} 条：{points}")
            else:
                print("[长期记忆] 本轮无可抽取的稳定画像，跳过入库")
        except Exception as e:
            print(f"[长期记忆] 画像抽取失败（不影响主流程）: {type(e).__name__}: {e}")

    task = asyncio.create_task(_job())
    _PROFILE_TASKS.add(task)
    task.add_done_callback(_PROFILE_TASKS.discard)
    return True

def _persist_turn_session(sid: str, session: dict, question: str, docs: list, qt: str):
    # 干什么：一轮结束后落库会话槽位——**守卫集中在这里**，避免 4 个调用点各写一份判断
    # 为什么守卫：澄清反问 / 能力边界话术 / 会话回顾都不是检索结果，写进去会污染 domain 与 subject
    if not sid:
        return
    if (qt or "").startswith(NON_RETRIEVAL_TYPES):
        return
    _save_session(sid, _extract_slots(question, docs or [], qt or "", session))

class SessionPrep(NamedTuple):
    # 干什么：两个 HTTP 端点共用的"本轮会话前置结论"，字段一一对应原先各自算一遍的局部变量
    session: dict
    followup: bool
    retrieval_query: str
    fill_note: str
    need_chat_meta: bool
    need_clarify: bool
    force_plan: bool

def _prepare_session(sid: str, question: str) -> SessionPrep:
    # 干什么：会话态加载 + 槽位填充 + 三个分支判定的**唯一实现**，给 /ask 与 /ask_stream 共用
    # 为什么抽出来：两个端点此前各写一份完全相同的 10 行判断，改一处漏一处；
    #              且下面三条判定有先后依赖（澄清优先于改写兜底），复制两份极易改歪
    session = _load_session(sid)
    followup = _is_followup(question)                            # 本轮是否依赖上文的追问
    retrieval_query, fill_note = _fill_slots(question, session)  # 本地槽位填充（检索用它，生成仍用原话）
    need_chat_meta = _is_chat_meta(question)                     # 会话元问题（"我们聊了什么"）
    # 指代型追问但会话里无线索 → 澄清反问（量化决策类除外：用户明确要结论时应直接声明能力边界）
    # 注意比较用 question.strip()：_fill_slots 返回的是 strip 过的查询，
    # 原问题带首尾空格时会误判成"已填充"，导致澄清/改写兜底双双失效
    need_clarify = bool(followup and not need_chat_meta and not _is_quant_request(question)
                        and not _has_context(session) and retrieval_query == question.strip())
    # 本地填充失败但会话有线索 → 允许规划 Agent 做 LLM 改写兜底
    force_plan = bool(followup and not need_clarify and not need_chat_meta
                      and retrieval_query == question.strip())
    return SessionPrep(session, followup, retrieval_query, fill_note,
                       need_chat_meta, need_clarify, force_plan)

def _session_state(prep: SessionPrep) -> dict:
    # 干什么：会话相关字段 → LangGraph 初始状态（两个端点共用，避免字段漏传导致行为不一致）
    return {
        "retrieval_query": prep.retrieval_query,       # 检索用查询（指代已消解/补全）
        "resolve_note": prep.fill_note,                # 指代说明 → 由 _context_hints 注入生成提示词
        "need_clarify": prep.need_clarify,             # 指代但无线索 → 澄清反问
        "need_chat_meta": prep.need_chat_meta,         # 会话元问题 → 会话摘要节点
        "session_snapshot": prep.session,              # 会话态快照（供会话摘要节点直接使用）
        "force_plan": prep.force_plan,                 # 填不出 → 允许规划做 LLM 改写
        "session_hint": _session_hint(prep.session),   # 会话线索（槽位+上一轮主题）
    }

def _session_payload(sid: str, prep: SessionPrep) -> dict:
    # 干什么：统一的"会话状态负载"，给 /ask 与 /ask_stream 的响应/事件复用（原来 4 处各拼一份）
    # 注：fill_note 与 resolve_note 同源（都是本地槽位填充的说明），前端两个字段都读，故都保留
    return {
        "enabled": bool(sid),
        "followup": prep.followup,
        "fill_note": prep.fill_note,
        "resolve_note": prep.fill_note,
        "retrieval_query": prep.retrieval_query,
        "slots": (prep.session or {}).get("slots") or {},
    }

def _context_hints(state: dict) -> str:
    # 干什么：把"指代消解说明 + 多跳检索说明"拼成统一提示片段，供各生成节点复用
    # 为什么统一：三类生成节点（fast_synth / analyzer / synthesizer）都要这段，三处各写易漂移
    parts = []
    if state.get("resolve_note"):
        parts.append(f"\n\n【指代消解】{state['resolve_note']}。请按此理解问题，回答仍面向用户原话。")
    if state.get("bridge_note"):
        parts.append(f"\n\n【多跳检索说明】{state['bridge_note']}")
    return "".join(parts)

def _docs_context(docs) -> str:
    # 干什么：把检索资料拼成提示词里的"资料区"（编号 + 来源标注），供生成节点复用
    # 为什么统一：analyzer 与 fast_synth 此前各写一份完全相同的格式化表达式，改一处必漏另一处
    return "\n\n".join(f"[资料{i + 1}]（来自 {m.get('source', '未知')}）\n{t}"
                       for i, (t, m) in enumerate(docs or []))

def _profile_hint(state: dict) -> str:
    # 干什么：用户画像提示片段（自带前导换行；无画像时返回空串，拼接后与不拼等价）
    text = "\n".join(f"- {p}" for p in (state.get("profile_hits") or []))
    return f"\n\n【用户画像，回答时适当结合】\n{text}" if text else ""

def _sse(obj: dict) -> str:
    # 干什么：把一个事件对象序列化成 SSE 帧（data: xxx + 空行结尾）
    # 为什么放在模块级：_replay 等模块级生成器也要用它（原来定义在 make_app 内，闭包看不见 → NameError）
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

def _replay(text: str):
    # 干什么：把静态答案按 40 字一段切片"回放"，模拟流式体验
    # 用于两处：① 缓存命中 ② 不调 LLM 的节点（not_supported / need_clarify / chat_meta）补推答案
    for piece in re.findall(r".{1,40}", text or ""):
        yield _sse({"type": "token", "text": piece})

async def _cache_put(question: str, answer: str, docs: list, qt: str, ns: str):
    # 干什么：统一写答案缓存（来源清单 / 问题类型 / 链路命名空间），避免多处逻辑漂移
    # 为什么 async：put 里要 encode 问题（同步约 19ms），用 to_thread 挪出事件循环
    if not answer:
        return
    await asyncio.to_thread(
        answer_cache.put, question, answer,
        [m.get("source") for _, m in (docs or [])],
        qt or _guess_query_type(question, docs or []), ns)

def _should_bridge(query: str, docs, conf: float = None, plan: dict = None) -> bool:
    # 干什么：桥接条件化——只在"确有必要"时才多花一次检索（必要时才调 LLM 定实体）
    # 判断顺序：① 是否多跳：优先用查询规划（query_planner）的 need_bridge 结论，
    #           规划缺失时退回本地关系词启发式；② 再要求"证据不足"（零命中 或 相似度 < 阈值）
    # 为什么以规划为准：规划 Agent 能看到整句语义（含子查询），比关键词表更准；
    #                  两者都不再并列判断，避免"既多意图又多跳"时的矛盾
    plan = plan or {}
    if plan:
        need = bool(plan.get("need_bridge"))
    else:
        need = _needs_bridge(query)
    if not need:
        return False                                           # 非多跳：绝不桥接
    if not docs:
        return True                                            # 首跳零命中：值得桥接
    if conf is None:
        return False                                           # 缺证据强度信号时保守不桥接
    return conf < BRIDGE_MIN_CONF                              # 证据太弱：桥接补证据

class AnswerCache:
    # 干什么：两级答案缓存——① 规范化问题精确匹配（落盘）② 语义近似匹配（内存向量）
    # 为什么：演示/答辩里重复提问很常见，命中后几乎零延迟
    def __init__(self, path):
        self.path = path
        self.exact = {}                 # 规范化问题 -> {q, answer, sources, query_type}
        self.vecs = {}                  # key -> 归一化向量（按 key 唯一，天然去重）
        self._load()

    def _norm(self, q: str) -> str:
        # 干什么：去掉空白与常见标点，让"chmod 755 是什么？"与"chmod755是什么"算同一个问题
        return re.sub(r"[\s，。？！、,.?!；;：:]", "", q.strip().lower())

    def _load(self):
        # 干什么：启动时读落盘缓存并重建语义索引（批量编码一次，之后查询只比向量）
        try:
            with open(self.path, encoding="utf-8") as f:
                self.exact = json.load(f)
        except Exception:
            self.exact = {}
        if len(self.exact) > CACHE_MAX_ENTRIES:                 # 与 put 的 FIFO 上限保持一致
            self.exact = dict(list(self.exact.items())[-CACHE_MAX_ENTRIES:])
        pairs = [(k, v["q"]) for k, v in self.exact.items() if v.get("q")]
        if not pairs:
            return                                              # 缓存为空：不做任何编码（启动不白等）
        vecs = EMBED_MODEL.encode([q for _k, q in pairs], normalize_embeddings=True)   # 一次批量
        self.vecs = {k: vecs[i] for i, (k, _q) in enumerate(pairs)}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.exact, f, ensure_ascii=False)
        except Exception as e:
            print(f"[缓存] 落盘失败（不影响回答）：{e}")

    def get(self, question: str, namespace: str = "fast"):
        # 返回：None 未命中；否则 {"hit": exact/semantic, "answer", "sources", "query_type"}
        # 参数 namespace：缓存命名空间（fast/full）——两条链路的答案不同，必须隔离，避免混用
        key = namespace + "::" + self._norm(question)
        if key in self.exact:
            return {"hit": "exact", **self.exact[key]}
        prefix = namespace + "::"
        keys = [k for k in self.vecs if k.startswith(prefix) and k in self.exact]
        if keys:
            qv = EMBED_MODEL.encode([question], normalize_embeddings=True)[0]
            # 向量已归一化，矩阵乘一次算出全部余弦相似度（原实现是 Python 逐条点积，随条数线性变慢）
            mat = np.asarray([self.vecs[k] for k in keys], dtype="float32")
            sims = mat @ np.asarray(qv, dtype="float32")
            j = int(np.argmax(sims))
            best = float(sims[j])
            if best >= CACHE_SEMANTIC_THRESHOLD:
                return {"hit": "semantic", "score": round(best, 3), **self.exact[keys[j]]}
        return None

    def put(self, question: str, answer: str, sources: list, query_type: str,
            namespace: str = "fast"):
        key = namespace + "::" + self._norm(question)
        self.exact[key] = {"q": question, "answer": answer,
                           "sources": sources or [], "query_type": query_type or ""}
        # 按 key 覆盖写：同一问题重复提问不会累积重复向量（原实现 append，vecs 只增不减）
        self.vecs[key] = EMBED_MODEL.encode([question], normalize_embeddings=True)[0]
        if len(self.exact) > CACHE_MAX_ENTRIES:      # 超限按插入顺序 FIFO 淘汰最旧一条
            oldest = next(iter(self.exact))
            self.exact.pop(oldest, None)
            self.vecs.pop(oldest, None)
        self._save()

answer_cache = AnswerCache(ANSWER_CACHE_PATH)

# ============================================================
# 11. LLM 模型与运行开关
# ============================================================
# openai-agents 0.18+ 会把模型名里的 "@cf/" 当 provider 前缀解析而报错，
# 所以显式构造 ChatCompletions 模型对象并绑定 AsyncOpenAI 客户端，绕过前缀解析
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
MODEL = OpenAIChatCompletionsModel(
    model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    openai_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        # 单次请求超时 30s（原 90s）：网关卡顿时快速失败，不让用户干等几十秒
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        # 不重试（原 1 次）：重试会让最坏耗时翻倍，宁可快速失败并让前端提示重试
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")),
    ),
)
HUMAN_REVIEW = os.getenv("HUMAN_REVIEW", "true").lower() != "false"   # 交互模式默认开人工确认
IS_TTY = sys.stdin.isatty()                                          # 非终端（管道/服务）自动跳过 input

# ============================================================
# 12. LangGraph 状态定义 + 五个 Agent
# ============================================================
class RAGState(TypedDict):
    query: str
    needs_retrieval: bool
    query_type: str
    docs: list
    analysis: str
    short_term: Annotated[list, add_messages]
    profile_hits: list
    event_log: Annotated[list, _add_logs]
    human_feedback: str
    answer: str
    fast_mode: bool          # True = 快速通道：跳过 router，检索后一次流式调用直接出答案
    skip_human_check: bool   # True = 本轮跳过人工确认（HTTP 端点置 True；替代原"切全局开关"，并发安全）
    bridge_note: str         # 多跳检索说明（中间实体、新增证据数），供生成阶段标注来源
    bridge_entity: str       # 桥接出的中间实体（人名/机构名等）
    retrieval_conf: float    # 首跳检索证据强度（top1 余弦相似度），用于决定是否桥接二跳
    retrieval_info: list     # 召回证据明细（顺序/来源/片段），前端展示"检索到了什么"
    retrieval_meta: dict     # 检索诊断（rerank 状态、top_sim 等）
    plan: dict               # 查询规划结果：{"sub_queries":[...], "need_bridge":bool, "bridge_entity":"..."}
    retrieval_query: str     # 实际用于检索的查询（可能被槽位填充/LLM 改写补全），生成阶段仍用 query
    need_clarify: bool       # 指代型追问且会话无线索 → 走澄清节点反问
    force_plan: bool         # 槽位填充失败但会话有线索 → 强制调用规划做 LLM 改写兜底
    session_hint: str        # 会话线索文本（槽位+上一轮主题），供规划 Agent 改写时参考
    resolve_note: str        # 指代消解说明（如「他」指『asd公司的创始人』），注入生成提示词
    need_chat_meta: bool     # 会话元问题（"我们聊了什么"）→ 走会话摘要节点
    session_snapshot: dict   # 会话态快照（供会话摘要节点直接使用）

SHARED_SYSTEM_PREFIX = (
    "你是用户的个人知识库问答助手。知识库为单一集合，内容包含：medical_kb/（健康医疗类笔记与科普"
    "资料），以及用户通过上传功能自行导入的任意主题文档（简历、笔记、学习资料、报告等，导入后自"
    "动入库可检索）。\n"
    "所有 Agent 共享以下工作原则：\n"
    "1. 回答必须基于提供的检索资料，禁止编造资料中未出现的内容。\n"
    "2. 使用中文回答，口语化、直接解决用户问题。\n"
    "3. 涉及操作、命令时给出具体可执行的示例。\n"
    "4. 若资料不足，明确说明并基于常识谨慎补充。\n"
    "5. 严格按要求的输出格式输出，不要多余解释。\n"
    "6. 【来源标注，必须遵守】每条关键结论后面标注来源：来自检索资料写「【知识库·文件名】」，"
    "来自模型自身知识写「【模型常识·未经核实】」。若知识库没有相关资料，必须明确写出「知识库未收录」，"
    "严禁把模型常识伪装成资料依据，也不要在没有资料支撑时给出精确数字或概率。\n\n"
    "你当前承担的专属角色如下：\n"
)
router_agent = Agent(
    name="router",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是路由助手。判断用户问题是否需要查阅个人知识库，并判定问题类型。知识库为单一集合："
        "内含健康医疗类笔记，以及用户自行导入的任意主题资料（学习资料、简历、笔记、文档等）。\n"
        "判定规则：\n"
        "1. 若问题要求给出【量化决策结论】——如成功率、录取概率、排名先后、打分评分、哪家更好、推荐哪家，"
        "一律判为 query_type=量化决策，needs_retrieval=false（系统会返回能力边界声明，不做生成）。\n"
        "2. 其余问题：除非明显是打招呼或纯闲聊，否则一律 needs_retrieval=true；拿不准时也设 true。\n"
        '只输出一行 JSON，格式：{"needs_retrieval": true/false, "query_type": "医疗健康|个人资料|学习资料|通用常识|量化决策"}。'
        "主题无法归入医疗健康时，若可能与用户导入的资料有关用 个人资料，否则用 通用常识。"
    ),
    model=MODEL,
)
analyzer_agent = Agent(
    name="analyzer",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是分析助手。基于提供的资料和【用户画像】【对话历史】做结构化分析草稿："
        "用要点列表提取关键事实，每条标注来源文件；不做最终润色。可结合用户画像让回答更贴合。"
    ),
    model=MODEL,
)
synthesizer_agent = Agent(
    name="synthesizer",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是总结助手。把分析草稿转化为面向用户的最终回答：口语化、给示例、考试相关标易错提醒。"
        "严格基于分析草稿，不编造。用中文。\n"
        "来源标注要求（本角色重点）：最终回答里每条关键结论都要带来源标记"
        "「【知识库·文件名】」或「【模型常识·未经核实】」；资料不足时在开头写明「知识库未收录」。"
    ),
    model=MODEL,
)
profile_extractor_agent = Agent(
    name="profile_extractor",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是用户画像抽取器。从用户问题和回答中抽取 0-3 条长期稳定的用户偏好/背景（中文短句，每行一条）。\n"
        "只抽跨会话有用的稳定信息，例如：身份/专业/职业、技术栈偏好、长期目标、明确的习惯偏好。\n"
        "一次性的问题内容、知识库里的领域知识、对话过程本身——这些都**不算**画像，不要抽。\n"
        "**没有可抽取内容时直接输出空（什么都不要输出），绝对不要输出任何解释或说明文字。**\n"
        "示例：输入「胸痛挂哪个科」→ 输出空；输入「我是软件工程专业的专升本学生，在找AI应用实习」"
        "→ 输出「用户是软件工程专业专升本学生，正在求职AI应用开发实习」"
    ),
    model=MODEL,
)
judge_agent = Agent(
    name="judge",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是 RAG 评测裁判。给定【问题】【检索资料】【生成答案】，对三项指标各打 0-1 分（1 最好）：\n"
        "context_precision = 检索资料与问题的相关程度\n"
        "faithfulness = 答案是否严格基于检索资料（不编造）\n"
        "answer_relevancy = 答案是否切题、有用\n"
        "只输出一行 JSON：{\"context_precision\":0.x,\"faithfulness\":0.x,\"answer_relevancy\":0.x}，不要多余文字。"
    ),
    model=MODEL,
)
# 快速通道 Agent：把"分析草稿 + 最终回答"压成一次调用，直接产出面向用户的答案
# 为什么单独建一个 Agent 而不是复用 synthesizer：两者提示词目标不同（这里要求省略分析过程、更长约束）
fast_agent = Agent(
    name="fast_answerer",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是回答助手（快速通道）。请直接输出面向用户的最终回答，省略分析过程："
        "条目化、口语化、给可执行示例；严格基于资料不编造；资料不足时明确说明。"
        "总长度控制在 400 字以内，不要重复问题。\n"
        "来源标注要求（本角色重点）：\n"
        "- 有资料依据的结论 → 标注「【知识库·文件名】」\n"
        "- 凭模型自身知识的补充 → 标注「【模型常识·未经核实】」，并提示用户可导入资料后复核\n"
        "- 资料中没有相关内容时，开头明确写「知识库未收录……」，再决定是否用模型常识作答"
    ),
    model=MODEL,
    model_settings=ModelSettings(max_tokens=ANSWER_MAX_TOKENS),   # 限制生成长度，缩短总耗时
)
# 多跳桥接 Agent：只负责"从问题与已有资料中定位中间实体"，输出极短 JSON
bridge_agent = Agent(
    name="bridge_extractor",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是实体桥接助手，用于多跳检索。给定用户问题和当前检索到的资料，找出回答问题所必需的关键中间实体"
        "（人名、机构名、产品名等，例如'某公司的创始人'需要先确定这个人是谁）。"
        '只输出一行 JSON：{"entity":"实体名，无法确定则空字符串","from":"kb"或"common"}，不要多余文字。'
        "资料中有依据时 from=kb；资料里没有、靠你自身知识给出候选用 from=common。"
    ),
    model=MODEL,
    model_settings=ModelSettings(max_tokens=80),    # 只要一个实体名，限制输出长度省时间
)
# 查询规划 Agent（原 planner + bridge_extractor 合并）：检索前一次性产出"检索计划"
# 为什么合并：两者本质是同一件事——"检索前怎么把查询变好"；拆两个会产生"既多意图又多年跳"的叠加歧义，
# 而合并后只需一次 LLM 往返（省 ~1s），且计划的三个字段由同一判断产生，不会互相矛盾
query_planner_agent = Agent(
    name="query_planner",
    instructions=SHARED_SYSTEM_PREFIX + (
        "你是查询规划助手，在检索前把用户问题变成更好的检索计划，输出三件事：\n"
        "1) sub_queries：若问题含多个彼此独立的提问意图（拼接词：顺便、另外、以及、还有、同时、分别、再帮我），"
        "**必须**拆成 2~3 个各自完整、能独立检索的子查询（保留科室/人群等限定词）；只有一个意图时把原问题原样放进数组。\n"
        "2) need_bridge：若问题属于'跨实体多跳'——答案对象本身不在问题里、需要先确定中间实体才能查它的属性，置为 true。\n"
        "3) bridge_entity：need_bridge 为 true 时给出中间实体的候选名（可用你自身知识给候选，不确定就留空字符串）。\n"
        "4) resolved_query：若问题含指代或省略（他/她/它/这个/那个，或省略了主语），且【上文线索】里能确定所指对象，"
        "把它补全成一句完整、可独立检索的问题；否则把原问题原样返回。\n"
        "示例：\n"
        '输入「chmod 755 是什么」→ {"sub_queries":["chmod 755 是什么"],"need_bridge":false,"bridge_entity":"","resolved_query":"chmod 755 是什么"}\n'
        '输入「OpenAI创始人是哪个大学毕业的」→ {"sub_queries":["OpenAI创始人是哪个大学毕业的"],"need_bridge":true,"bridge_entity":"Sam Altman","resolved_query":"OpenAI创始人是哪个大学毕业的"}\n'
        '输入「胸痛挂哪个科顺便帮我查查asd公司创始人的代步工具是什么」→ '
        '{"sub_queries":["胸痛挂哪个科","asd公司创始人的代步工具是什么"],"need_bridge":true,"bridge_entity":"","resolved_query":"胸痛挂哪个科顺便帮我查查asd公司创始人的代步工具是什么"}\n'
        '输入「他是什么专业的」（上文线索：上文讨论对象=dsa）→ '
        '{"sub_queries":["dsa 是什么专业的"],"need_bridge":false,"bridge_entity":"","resolved_query":"dsa 是什么专业的"}\n'
        '只输出一行 JSON：{"sub_queries":["..."],"need_bridge":true/false,"bridge_entity":"...","resolved_query":"..."}，不要多余文字。'
    ),
    model=MODEL,
    model_settings=ModelSettings(max_tokens=256),   # 只要短 JSON，限制输出省时间
)

# ============================================================
# 13. 图节点
# ============================================================
async def router_node(state: RAGState) -> dict:
    # 干什么：让 LLM 判断是否需要检索、以及问题主题类型（结果直接决定后续走向）
    t0 = time.perf_counter()                                   # 计时起点：本节点耗时用于性能观测
    def _msg_role(m):
        return getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)

    def _msg_content(m):
        return getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
    history_text = "\n".join(
        f"{('用户' if _msg_role(m) == 'user' else '助手')}：{_msg_content(m)}"
        for m in (state.get("short_term") or [])[-6:]          # 只带最近 3 轮，控制上下文长度
    )
    profile_text = "\n".join(f"- {p}" for p in (state.get("profile_hits") or []))
    ctx = ""
    if history_text:
        ctx += f"\n【最近对话】\n{history_text}"
    if profile_text:
        ctx += f"\n【已知用户画像】\n{profile_text}"
    res = await Runner.run(router_agent, f"{state['query']}{ctx}")
    parsed = _parse_json_output(res.final_output, {"needs_retrieval": True, "query_type": "通用常识"})
    need = bool(parsed.get("needs_retrieval", True))
    qt = parsed.get("query_type", "通用常识")
    print(f"[router] LLM 判断 needs_retrieval={need}, query_type={qt}")
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": "router", "ms": ms})      # 前端展示各节点耗时
    rec = event_logger.log("router", f"query={state['query']}", f"need={need},type={qt}")
    return {"needs_retrieval": need, "query_type": qt, "event_log": [rec]}

async def retriever_node(state: RAGState) -> dict:
    # 干什么：① 多意图问题先做"查询分解"（一次 planner 调用拆成 2~3 个子查询）；
    #         ② 对每个（子）查询做混合检索；③ 合并去重后按链路取前若干块
    # 为什么分解：一个向量/一个 BM25 词袋同时表达两个意图会互相稀释，
    #            实测双意图拼接时正确段从 RRF #3 掉到 #5 —— 分解后各意图互不干扰
    # 快速通道（fast_mode）下 query_type 为空，会走"两库都查"分支，等效于免路由检索
    t0 = time.perf_counter()
    qt = state.get("query_type", "通用常识")
    q = state["query"]                                         # 原始问题：生成阶段用它（保持用户原话）
    rq = (state.get("retrieval_query") or q).strip()           # 检索问题：可能已被槽位填充/LLM 改写补全
    # ---- ① 查询规划（合并原 planner + bridge_extractor）：只在需要时才多花一次 LLM ----
    # 触发条件：多意图（拼接词/多疑问点）或 疑似多跳（含强关系词）或 槽位填充失败需改写兜底
    plan, subs = {}, [rq]
    if _needs_decompose(rq) or _needs_bridge(rq) or state.get("force_plan"):
        prompt = f"用户问题：{rq}"
        hint = state.get("session_hint") or ""
        if hint:
            # 会话线索（槽位+上一轮主题）交给规划 Agent，供"指代改写兜底"使用
            prompt += f"\n\n【上文线索】{hint}\n若上面的问题含指代或省略（如'他/它/这个'），请用上文线索把 resolved_query 补成完整问题。"
        res = await Runner.run(query_planner_agent, prompt)
        # 兜底值必须是"规划形状"的非空字典：_should_bridge 用 `if plan:` 判"本轮有没有规划"，
        # 空字典会被当成"没有规划"而退回关键词启发式；解析失败时按"单查询、不桥接"兜底
        # （原版兜的是 router 形状的 dict，字段对不上 planner 的消费方，纯靠巧合不报错）
        plan = _parse_json_output(res.final_output,
                                  {"sub_queries": [rq], "need_bridge": False,
                                   "bridge_entity": "", "resolved_query": rq})
        cand_subs = [str(s).strip() for s in (plan.get("sub_queries") or []) if str(s).strip()]
        if 1 < len(cand_subs) <= 3:                            # 上限 3 条，防止延迟被放大
            subs = cand_subs
            rq = subs[0]                                       # 子查询列表存在时，后续按子查询分别检索
        resolved = str(plan.get("resolved_query") or "").strip()
        if resolved and len(subs) == 1 and resolved != rq:
            rq, subs = resolved, [resolved]                    # LLM 改写兜底：指代替换为上文实体
            plan["rewritten"] = True
        _emit({"type": "plan", "subs": subs,
               "need_bridge": bool(plan.get("need_bridge")),
               "entity": str(plan.get("bridge_entity") or ""),
               "retrieval_query": rq, "rewritten": bool(plan.get("rewritten"))})
        print(f"[query_planner] 子查询={subs} | need_bridge={bool(plan.get('need_bridge'))} "
              f"| entity={plan.get('bridge_entity')} | 检索用查询={rq}")
    # ---- ② 每个子查询"独立召回 + 独立精排"，各自保留 topN（分组配额）----
    # 关键点：不能把所有子查询的结果混在一起再统一截断——那样先检索的子查询会把名额占满，
    # 后面的子查询证据全被丢掉（这正是上一版的缺陷）。改为分组保留、再交错合并。
    per_sub, groups, conf = RETRIEVE_PER_SUB, [], 1.0
    sub_pool = MULTI_SUB_POOL if len(subs) > 1 else None      # 多子查询时用小池子控制总耗时
    per_sub_stats = []                                        # 每个子查询独立的诊断 dict（共享一个会被后面的覆盖，前端只见最后一条）
    for sq in subs:
        # 单库架构：所有问题查同一个知识库（原"医疗/考试"分库逻辑已随合并删除）
        st_sq = {}                                            # 本子查询的检索诊断：rerank 是否执行/降级/跳过
        # to_thread：检索里的本地嵌入(encode)+chroma 查询+BM25 都是同步 CPU 活（实测 encode 单条约 19ms），
        # 直接在 async 节点里跑会阻塞事件循环、并发请求互相排队；丢进线程池后多个请求可真正并行
        hits_sq = await asyncio.to_thread(retrieve, sq, per_sub, st_sq, sub_pool)
        # v2 修复①：证据强度直接复用检索内部已算出的 top_sim——原版用 _retrieval_confidence
        # 把同一个查询再编码一次、再查一遍 chroma，每个子查询白多一次 encode + 一次向量查询
        # v2 修复②：取各子查询的**最小值**——任一子查询证据不足就该桥接补证据；
        # 原版取 max 会拿强意图的 conf 掩盖弱意图，多跳子查询永远等不来桥接
        conf = min(conf, float(st_sq.get("top_sim") or 0.0))
        per_sub_stats.append(st_sq)
        groups.append(hits_sq[:per_sub])                      # 每组只保留自己 topN
    # 诊断聚合（供打印/审计/前端）：rerank 状态拼接去重，top_sim 与 conf 同源（最小值）
    stats = {"rerank": "/".join(dict.fromkeys(s.get("rerank", "n/a") for s in per_sub_stats)) or "n/a",
             "top_sim": conf, "per_sub_stats": per_sub_stats}
    # ---- ③ 交错合并：先取每组的第 1 名，再取每组的第 2 名…… 保证每个意图都有代表证据 ----
    merged, seen = [], set()
    for rank in range(per_sub):
        for gi, g in enumerate(groups):
            if rank >= len(g):
                continue
            t, m = g[rank]
            key = (t[:80], m.get("source"))                   # 前 80 字 + 来源去重
            if key in seen:
                continue
            seen.add(key)
            merged.append((t, m, gi))                         # gi = 来自第几个子查询（用于展示分组）
    total_limit = per_sub if len(subs) == 1 else min(per_sub * len(subs), RETRIEVE_MAX_TOTAL)
    merged = merged[:total_limit]
    hits = [(t, m) for t, m, _ in merged]
    # 同源相邻块扩展：命中块的同一文件邻居补进上下文（治"整文档切多块、中间块漏召回"）
    hits = _expand_same_source(hits, MED_KB)
    print(f"[retriever]（知识库={qt}｜子查询 {len(subs)} 个）分组配额 {per_sub}/组 → 合并 {len(hits)} 块 | "
          f"rerank={stats.get('rerank', 'n/a')} | 证据强度={conf}")
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": "retriever", "ms": ms})
    # 召回证据明细：以扩展后的 hits 为准（否则相邻扩展块在界面上看不到）
    # gi 通过指纹回查——扩展块不属于任何子查询，归入其锚点所在组（查不到记 1）
    gi_of = {(t[:80], m.get("source")): gi for t, m, gi in merged}
    info = [{"rank": i + 1, "sub": gi_of.get((t[:80], m.get("source")), 0) + 1,
             "expanded": bool(m.get("expanded")), "source": m.get("source", "?"),
             "snippet": ("（相邻扩展）" if m.get("expanded") else "") + t[:46].replace("\n", " ")}
            for i, (t, m) in enumerate(hits)]
    rec = event_logger.log("retriever", f"query={q}",
                           f"子查询 {len(subs)} / 分组配额 {per_sub} / 合并 {len(hits)} 块 / "
                           f"rerank={stats.get('rerank', 'n/a')} / conf={conf}")
    return {"docs": hits, "retrieval_conf": conf, "retrieval_info": info, "plan": plan,
            "retrieval_meta": dict(stats, top_sim=conf, ms=ms, subs=subs,
                                   per_sub=per_sub, groups=[len(g) for g in groups]),
            "event_log": [rec]}

async def analyzer_node(state: RAGState) -> dict:
    # 干什么：基于检索资料做结构化分析草稿（不润色，润色交给 synthesizer）
    t0 = time.perf_counter()
    docs = _truncate_docs(state.get("docs") or [])             # 截断资料块，控制 prefill 长度
    qt = state.get("query_type", "通用常识")
    if docs:
        # 按问题类型追加针对性要求（医疗类标安全提示且不给处方剂量）
        extra = ("（注意：这是医疗类问题，分析时额外标注安全注意事项与及时就医提示，且不给出具体处方药剂量）"
                 if qt in ("医疗", "医疗健康") else "")
        prompt = (f"基于以下资料做结构化分析草稿{extra}：\n\n"
                  f"{_docs_context(docs)}\n\n用户问题：{state['query']}")
    else:
        prompt = f"以下问题无知识库资料，用常识做简要分析草稿：{state['query']}"
    prompt += _profile_hint(state) + _context_hints(state)     # 画像 + 指代消解/多跳说明（统一拼装）
    print(f"[analyzer] 基于 {'检索到的 ' + str(len(docs)) + ' 块资料' if docs else '常识'} 做结构化分析...")
    result = await Runner.run(analyzer_agent, prompt)
    analysis = result.final_output
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": "analyzer", "ms": ms})
    rec = event_logger.log("analyzer", f"资料 {len(docs)} 块/画像 {len(state.get('profile_hits') or [])} 条",
                           f"分析草稿 {len(analysis)} 字")
    return {"analysis": analysis, "event_log": [rec]}

async def human_check_node(state: RAGState) -> dict:
    # 干什么：人机协同节点——用 interrupt 暂停图，把草稿交给人确认后再继续
    # v2 修复：跳过确认改读本轮 state（skip_human_check）。原来三个端点都是
    # "保存全局 HUMAN_REVIEW → 改 False → finally 恢复"，并发请求会互相覆盖恢复值，
    # 导致人工确认被静默永久关闭（A 恢复 True 后 B 又把 False 写回去）
    if not HUMAN_REVIEW or state.get("skip_human_check"):
        return {"human_feedback": "y"}
    payload = {
        "question": state["query"],
        "analysis": state.get("analysis", ""),
        "docs_summary": [f"[资料{i + 1}]({m.get('source', '?')}) {t[:120]}" for i, (t, m) in enumerate(state.get("docs") or [])],
    }
    feedback = interrupt(payload)
    return {"human_feedback": feedback}

async def synthesizer_node(state: RAGState) -> dict:
    # 干什么：把分析草稿转成面向用户的最终回答；有修改意见时按其调整
    t0 = time.perf_counter()
    analysis = state.get("analysis") or ""
    fb = state.get("human_feedback") or ""
    prompt = (
        f"用户问题：{state['query']}\n\n分析草稿：\n{analysis}\n\n"
        f"请把上面的分析草稿转化为面向用户的最终回答。"
    )
    if fb and fb.lower() not in ("y", "yes", "是", "true", ""):
        prompt += f"\n\n【人工修改意见，请据此调整最终回答】：{fb}"
    prompt += _context_hints(state)                            # 指代消解/多跳说明（统一拼装）
    # 走流式调用：token 实时进 custom 流，/ask_stream 才能边生成边推给浏览器
    answer = await _run_agent_streamed(synthesizer_agent, prompt)
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": "synthesizer", "ms": ms})
    rec = event_logger.log("synthesizer", f"问题={state['query']}", f"答案 {len(answer)} 字")
    return {
        "answer": answer,
        "short_term": [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": answer},
        ],
        "event_log": [rec],
    }

CLARIFY_ANSWER = (
    "我还不知道你说的「他/它/这个」具体指什么——**这是本轮对话里第一次出现该指代**，"
    "会话里也没有可对应的对象（没有上文槽位，也没有上一轮主题）。\n\n"
    "请把对象直接说清楚，例如：\n"
    "- 「dsa 是哪个专业的」\n"
    "- 「胸痛需要拍片吗」\n"
    "- 「asd 公司的创始人是谁」\n\n"
    "说明：系统采用轻量会话记忆（只保存结构化槽位与上一轮主题，不保存答案原文），"
    "所以跨轮指代需要上一轮先提到过那个对象。"
)

def _no_llm_payload(state: RAGState, node: str, answer: str, query_type: str,
                    note: str, t0: float) -> dict:
    # 干什么：三个"不调 LLM、直接返回话术"节点的统一返回体
    # 为什么统一：chat_meta / need_clarify / not_supported 除话术与标签外完全同构，
    #            此前各写一份 ~18 行样板（含 timing 事件 + 审计记录 + short_term 拼装），易漂移
    # 参数 t0：节点各自的计时起点，由调用方传入，保证 timing 事件仍覆盖该节点的完整耗时
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": node, "ms": ms})          # 前端各节点耗时展示
    rec = event_logger.log(node, f"query={state['query']}", note)
    return {
        "answer": answer,
        "query_type": query_type,
        "short_term": [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": answer},
        ],
        "event_log": [rec],
    }

async def chat_meta_node(state: RAGState) -> dict:
    # 干什么：会话元问题节点（如"我们聊了什么"）——直接基于会话态生成回答，不检索、不调 LLM
    # 为什么需要：这类问题的答案在会话里而不在知识库里，走检索必然答"未收录"（实测问题）
    # 说明：会话态只保存"结构化槽位 + 最近 3 轮主题"，不保存答案原文，所以只能答"聊过什么主题"
    t0 = time.perf_counter()
    session = state.get("session_snapshot") or {}
    recent = session.get("recent") or []
    slots = session.get("slots") or {}
    if not recent:
        answer = ("这是本会话的**第一个问题**，之前没有对话记录。\n\n"
                  "说明：系统采用轻量会话记忆——只保存结构化槽位（讨论对象/主题）与最近几轮的主题，"
                  "**不保存答案原文**。")
    else:
        lines = []
        for i, r in enumerate(recent):
            q_prev = str(r.get("q") or "")
            subj = str(r.get("subject") or "")
            lines.append(f"{i + 1}. {q_prev}" + (f"（主题：{subj}）" if subj else ""))
        answer = "本会话最近几轮：\n" + "\n".join(lines)
        if slots.get("subject"):
            answer += f"\n\n当前讨论对象：**{slots['subject']}**"
        if slots.get("domain"):
            answer += f"　主题领域：{slots['domain']}"
        answer += ("\n\n说明：系统只保存结构化槽位与每轮主题，**不保存答案原文**；"
                   "如果你要追问某一条，直接说对象即可（例如「胸痛需要拍片吗」）。")
    return _no_llm_payload(state, "chat_meta", answer, "会话回顾",
                           "会话元问题 → 返回会话摘要（不检索）", t0)

async def need_clarify_node(state: RAGState) -> dict:
    # 干什么：澄清节点——指代型追问但会话里无线索时，直接反问澄清，不检索、不调 LLM
    # 为什么需要：无上下文时硬答只会罗列所有候选（实测问题），先反问既准确又省钱
    # 注意：本节点由端点预先判定（state.need_clarify），START 条件边直接短路到这里
    return _no_llm_payload(state, "need_clarify", CLARIFY_ANSWER, "需要澄清",
                           "指代型追问但会话无线索 → 反问澄清", time.perf_counter())

async def not_supported_node(state: RAGState) -> dict:
    # 干什么：能力边界节点——判定为量化决策类问题时直接返回固定话术，不调用 LLM 生成
    # 为什么不生成：没有结构化数据支撑时让模型"编"成功率是不负责任的，声明边界比编造更可信
    return _no_llm_payload(state, "not_supported", NOT_SUPPORTED_ANSWER, "量化决策（暂不支持）",
                           "能力边界：量化决策类，返回固定话术", time.perf_counter())

async def bridge_retrieve_node(state: RAGState) -> dict:
    # 干什么：多跳兜底节点——先确定问题所需的中间实体（如"DeepSeek 创始人"→梁文锋），
    #         再用"实体 + 原问题"做第二跳检索，把跨文档证据补齐
    # 为什么需要：像"X 的创始人是哪个大学毕业的"这类问题，一次检索很难同时命中实体本身和它的属性，
    #             单跳检索只能拿到一半证据，导致回答缺依据
    # 参数 state：含 query 与第一跳检索到的 docs
    # 返回：合并去重后的 docs + bridge_note（供生成阶段标注来源）
    t0 = time.perf_counter()
    q = state["query"]
    docs = list(state.get("docs") or [])
    plan = state.get("plan") or {}
    # ---- 1) 优先用查询规划给出的中间实体（省掉一次 LLM 调用）；规划没给才现场抽取 ----
    ent = str(plan.get("bridge_entity") or "").strip()
    if ent:
        from_kb = False                                        # 规划实体源自模型知识，未经知识库核实
        note_src = "查询规划（模型常识·未经核实）"
    else:
        doc_ctx = "\n\n".join(f"[{m.get('source', '?')}]\n{t[:400]}" for t, m in docs[:3]) or "（当前没有检索到资料）"
        res = await Runner.run(bridge_agent, f"问题：{q}\n\n当前资料：\n{doc_ctx}")
        parsed = _parse_json_output(res.final_output, {})        # 只做 .get()，兜空字典即可
        ent = str(parsed.get("entity") or "").strip()
        from_kb = parsed.get("from") == "kb"
        note_src = "知识库资料" if from_kb else "模型常识·未经核实"
    # ---- 2) 用实体做第二跳检索，与第一跳结果合并去重 ----
    note, added = "未能确定中间实体，本次只用第一跳证据作答", 0
    if ent:
        q2 = f"{ent} {q}"                                      # 实体 + 原问题，比只查实体更聚焦
        hits2 = await asyncio.to_thread(_hybrid_retrieve, q2, MED_KB, 3)   # 同 retriever：编码+检索挪出事件循环
        seen = {(t[:80], m.get("source")) for t, m in docs}    # 前 80 字 + 来源作为去重键
        for t, m in hits2:
            key = (t[:80], m.get("source"))
            if key not in seen:
                seen.add(key)
                docs.append((t, m))
                added += 1
        docs = docs[:6]                                        # 控制上下文长度，别把 prompt 撑爆
        note = f"中间实体「{ent}」（依据：{note_src}），已用它做第二跳检索，新增 {added} 块证据"
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": "bridge_retrieve", "ms": ms})
    print(f"[bridge_retrieve] {note}")
    rec = event_logger.log("bridge_retrieve", f"query={q}", note[:200])
    # 桥接后重新汇报证据清单（含第二跳新增的块），前端展示"两跳合并后的召回结果"
    info = [{"rank": i + 1, "source": m.get("source", "?"), "snippet": t[:46].replace("\n", " ")}
            for i, (t, m) in enumerate(docs)]
    return {"docs": docs, "bridge_entity": ent, "bridge_note": note,
            "retrieval_info": info, "retrieval_meta": {"stage": "bridge", "added": added, "ms": ms},
            "event_log": [rec]}

async def fast_synth_node(state: RAGState) -> dict:
    # 干什么：快速通道节点——"分析 + 总结"压成一次流式 LLM 调用，直接产出最终回答
    # 为什么快：省掉 analyzer 与 synthesizer 两次串行往返，首字延迟大幅提前
    # 代价：没有独立的分析草稿，因此不支持人工确认（HITL）；交互模式仍走三段式
    t0 = time.perf_counter()
    docs = _truncate_docs(state.get("docs") or [])             # 资料截断，缩短 prefill
    if docs:
        prompt = (f"请基于以下资料回答问题，直接给出最终回答：\n\n"
                  f"{_docs_context(docs)}\n\n用户问题：{state['query']}")
    else:
        prompt = f"知识库暂无相关资料，请用常识简要回答：{state['query']}"
    prompt += _profile_hint(state) + _context_hints(state)     # 画像 + 指代消解/多跳说明（统一拼装）
    answer = await _run_agent_streamed(fast_agent, prompt)     # token 实时进 custom 流
    ms = int((time.perf_counter() - t0) * 1000)
    _emit({"type": "timing", "node": "fast_synth", "ms": ms})
    rec = event_logger.log("fast_synth", f"资料 {len(docs)} 块", f"答案 {len(answer)} 字 / {ms}ms")
    return {
        "answer": answer,
        "short_term": [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": answer},
        ],
        "event_log": [rec],
    }

def _parse_json_output(text, default):
    # 干什么：容错解析 Agent 的 JSON 输出（模型偶尔带前后缀文字）——router / planner / bridge / judge 共用
    # 参数 default：解析失败时的兜底值。router 必须兜"需检索"，只做 .get() 的调用方兜空字典即可
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return default

def route_decision(state: RAGState) -> str:
    # 干什么：router 之后的条件边——量化决策类短路到能力边界节点，需要检索就去 retriever，否则直奔 analyzer
    if state.get("query_type") == "量化决策":
        return "not_supported"
    return "retriever" if state["needs_retrieval"] else "analyzer"

def start_decision(state: RAGState) -> str:
    # 干什么：入口分支（顺序即优先级）——① 会话元问题 → 会话摘要；② 量化决策 → 能力边界声明；
    #         ③ 指代型追问但无线索 → 澄清反问；④ 快速通道直接进检索；⑤ 普通流程先路由
    # 为什么"量化决策"要排在"澄清"之前：用户明确要一个打分结论，应当直接声明不支持，
    #                                      而不是先反问"他指谁"（实测踩过这个顺序坑）
    if state.get("need_chat_meta"):
        return "chat_meta"
    if _is_quant_request(state.get("query", "")):
        return "not_supported"
    if state.get("need_clarify"):
        return "need_clarify"
    return "retriever" if state.get("fast_mode") else "router"

def after_retriever(state: RAGState) -> str:
    # 干什么：检索之后的分支——① 多跳且证据不足时做桥接二跳（结论来自查询规划，非关键词）；
    #         ② 快速通道一次流式出答案；③ 普通流程走 analyzer → human_check → synthesizer
    if _should_bridge(state.get("query", ""), state.get("docs") or [],
                      state.get("retrieval_conf"), state.get("plan")):
        return "bridge_retrieve"
    return "fast_synth" if state.get("fast_mode") else "analyzer"

def after_bridge(state: RAGState) -> str:
    # 干什么：桥接检索之后同样按链路分流（快速出答案 / 走完整流程）
    return "fast_synth" if state.get("fast_mode") else "analyzer"

async def _run_agent_streamed(agent, prompt: str) -> str:
    # 干什么：跑一个 agent，并把生成的 token 实时推给 LangGraph 的 custom 流（SSE 用）
    # 为什么：Runner.run 要等整段生成完才返回；SSE 需要边生成边推，所以用 run_streamed
    # 参数 agent：openai-agents 的 Agent 对象；prompt：输入文本
    # 返回：完整回答文本（同时已通过 writer 推送过 token 事件）
    from openai.types.responses import ResponseTextDeltaEvent
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()          # 非流式调用上下文里是 no-op，安全
    except Exception:
        writer = None                         # 个别版本在非流式上下文会抛异常，兜底为 None
    result = Runner.run_streamed(agent, prompt)      # 流式运行（同步返回句柄，事件异步迭代）
    text_parts = []
    async for ev in result.stream_events():
        # raw_response_event = 底层模型的原始事件流，从中只挑文本增量
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            delta = ev.data.delta or ""
            if not delta:
                continue
            text_parts.append(delta)
            if writer is not None:
                writer({"type": "token", "text": delta})     # 每个增量立刻推给前端
    text = "".join(text_parts).strip()
    return text or (result.final_output or "")             # 兜底：万一没收到增量就用最终输出

# ---- 14. 状态图装配 ----
builder = StateGraph(RAGState)
builder.add_node("router", router_node)
builder.add_node("retriever", retriever_node)
builder.add_node("analyzer", analyzer_node)
builder.add_node("human_check", human_check_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("fast_synth", fast_synth_node)          # 快速通道：分析+总结合一
builder.add_node("not_supported", not_supported_node)    # 能力边界：量化决策类直接返回话术
builder.add_node("need_clarify", need_clarify_node)      # 会话澄清：指代型追问但无线索时反问
builder.add_node("chat_meta", chat_meta_node)            # 会话回顾：回答"我们聊了什么"这类元问题
builder.add_node("bridge_retrieve", bridge_retrieve_node)  # 多跳兜底：实体桥接二跳检索
# 入口分支：会话元问题 → chat_meta；量化决策 → not_supported；指代无线索 → need_clarify；
#          fast_mode 直接检索；否则先走 router
builder.add_conditional_edges(START, start_decision,
                             {"not_supported": "not_supported", "need_clarify": "need_clarify",
                              "chat_meta": "chat_meta", "retriever": "retriever", "router": "router"})
builder.add_conditional_edges("router", route_decision,
                             {"not_supported": "not_supported", "retriever": "retriever", "analyzer": "analyzer"})
# 检索后分支：需要多跳先桥接，否则按链路直达（快速出答案 / 完整流程分析）
builder.add_conditional_edges("retriever", after_retriever,
                             {"bridge_retrieve": "bridge_retrieve", "fast_synth": "fast_synth", "analyzer": "analyzer"})
builder.add_conditional_edges("bridge_retrieve", after_bridge,
                             {"fast_synth": "fast_synth", "analyzer": "analyzer"})
builder.add_edge("not_supported", END)
builder.add_edge("need_clarify", END)
builder.add_edge("chat_meta", END)
builder.add_edge("fast_synth", END)
builder.add_edge("analyzer", "human_check")
builder.add_edge("human_check", "synthesizer")
builder.add_edge("synthesizer", END)
graph = builder.compile(checkpointer=MemorySaver())              # MemorySaver = 短期记忆（按 thread_id 隔离）
print("[LangGraph] 状态图已编译：START →(fast)retriever → fast_synth → END ／ START → router →(retriever)→ analyzer → human_check → synthesizer → END")

def _drop_thread(config: dict):
    # 干什么：一轮跑完后删掉本轮的 checkpoint
    # 为什么必须删：InMemorySaver 只写不回收，每轮 checkpoint 里都带着 docs 原文（实测单轮 10~19 块）；
    #              每问一次新建一个 thread_id，长跑服务内存单调增长。interrupt 恢复需要 checkpointer，
    #              所以不能干脆不挂，只能"用完即删"。删除不存在的 thread 是 no-op，天然安全
    tid = ((config or {}).get("configurable") or {}).get("thread_id")
    if not tid:
        return
    try:
        graph.checkpointer.delete_thread(tid)
    except Exception as e:
        print(f"[记忆] checkpoint 回收失败（不影响回答）：{type(e).__name__}: {e}")

async def run_turn(q: str, profile_hits: list, short_term_history: list,
                   fast_mode: bool = False, extra_state: dict = None,
                   skip_human_check: bool = False) -> dict:
    # 干什么：跑一轮完整问答；遇到 interrupt 就打印草稿、等人工确认后 resume
    # 参数 fast_mode：True 走快速通道（跳过 router、分析+总结合一），Web 接口默认用它
    # 参数 extra_state：附加状态（会话槽位填充结果等，如 retrieval_query/need_clarify/session_hint）
    # 参数 skip_human_check：True 时本轮不进人工确认（HTTP 端点用；原版靠切全局开关，并发会互相覆盖）
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}   # 每轮独立 thread，避免串记忆
    state_in = {
        "query": q, "needs_retrieval": False, "query_type": "",
        "docs": [], "analysis": "", "short_term": short_term_history,
        "profile_hits": profile_hits, "event_log": [], "human_feedback": "", "answer": "",
        "fast_mode": fast_mode,
        "skip_human_check": skip_human_check,
    }
    if extra_state:
        state_in.update(extra_state)                              # 会话相关字段并入初始状态
    try:
        result = await graph.ainvoke(state_in, config=config)
        while isinstance(result, dict) and result.get("__interrupt__"):
            intr = result["__interrupt__"][0]
            payload = intr.value
            print("\n【人工确认】分析草稿预览：")
            print(payload.get("analysis", ""))
            for d in payload.get("docs_summary", []):
                print(f"  · {d}")
            if IS_TTY:
                fb = input("确认发布回答? [y 直接发布 / 输入修改意见]: ").strip()
            else:
                fb = "y"                                             # 服务/管道模式自动放行，避免阻塞
            result = await graph.ainvoke(Command(resume=fb), config=config)
        return result
    finally:
        _drop_thread(config)                                      # 本轮 checkpoint 用完即删，防内存无限增长

# ============================================================
# 15. LLM-as-judge 评测
# ============================================================
EVAL_SET = [
    # 内置评测集：每条 = 一个测试问题 + ref 参考答案要点（给人工核对用，judge 只看检索资料与生成答案）
    # 前 5 条考 Linux 命令类资料（学习资料域），后 2 条考医疗急救/慢病管理（医疗健康域），
    # 覆盖两类主题是为了验证"单库混合检索"在跨主题语料上都能召回正确段落
    {"q": "怎么用tar命令压缩文件", "ref": "tar -czvf 打包压缩；-xzvf 解压；f 必须放最后紧跟文件名"},
    {"q": "chmod 755 是什么意思", "ref": "r=4 w=2 x=1；755=主人rwx/同组r-x/其他r-x"},
    {"q": "普通用户可以用 chown 把文件属主改成 root 吗", "ref": "不能，改属主成 root 需 root 权限，普通用户会被拒绝"},
    {"q": "sudo 和 su 的区别", "ref": "sudo 临时借 root 权限执行单条命令；su 切换整个会话为 root"},
    {"q": "怎么递归修改目录权限", "ref": "chmod -R 递归；如 chmod -R 755 dir"},
    {"q": "突发口角歪斜肢体无力是怎么回事", "ref": "疑似脑卒中，立即急诊神经内科，强调时间窗内溶栓"},
    {"q": "糖尿病患者饮食要注意什么", "ref": "控制总热量、低GI主食、低盐、定时定量、监测血糖"},
]

async def eval_collect():
    # 干什么：对内置测试集逐题跑完整问答流程，再让 judge Agent 按三项指标各打 0-1 分
    # 为什么自己写而不用 Ragas：零额外依赖、指标口径可解释（judge 提示词里写死了三项含义），
    # 论文/答辩场景里"怎么算的分"比"分数多高"更重要
    # 返回：(rows 每题明细列表, avg 三项平均分字典)
    rows = []                                                    # 每题一行：问题 + 三项得分
    for item in EVAL_SET:
        # 先召回与本题相关的长期画像（与真实问答路径一致，保证评测覆盖画像注入分支）
        ph = await asyncio.to_thread(profile_memory.recall, item["q"], 3)
        # 跑完整流程出答案；skip_human_check=True：评测是无人值守的批量调用，
        # 不能进 interrupt 等人工确认（原版靠切全局开关，并发请求会互相覆盖）
        res = await run_turn(item["q"], ph, [], skip_human_check=True)
        docs = res.get("docs") or []                             # 本题实际检索到的资料块
        answer = res.get("answer") or ""                         # 本题生成的最终答案
        # 拼接检索资料给 judge；与生成路径同样按 DOC_CHAR_LIMIT 截断，控制 prefill 长度
        ctx = "\n".join(t[:DOC_CHAR_LIMIT] for t, m in docs)
        # judge 只看"问题 + 检索资料 + 生成答案"三样，不给参考答案——评的是 RAG 链路本身
        jr = await Runner.run(judge_agent,
                              f"问题：{item['q']}\n检索资料：{ctx}\n生成答案：{answer}")
        s = _parse_json_output(jr.final_output, {})              # 容错解析 JSON；失败兜空 dict
        # 逐项取分：`or 0` 防 judge 输出 null/缺字段时 float(None) 抛异常
        cp = float(s.get("context_precision", 0) or 0)           # 检索资料与问题的相关度
        fa = float(s.get("faithfulness", 0) or 0)                # 答案是否严格基于资料（不编造）
        ar = float(s.get("answer_relevancy", 0) or 0)            # 答案是否切题、有用
        rows.append({"q": item["q"], "context_precision": cp, "faithfulness": fa, "answer_relevancy": ar})
    n = len(rows)
    # 三项平均分：论文里报的"系统整体 RAG 质量"就是这三个数
    avg = {"context_precision": sum(r["context_precision"] for r in rows) / n,
           "faithfulness": sum(r["faithfulness"] for r in rows) / n,
           "answer_relevancy": sum(r["answer_relevancy"] for r in rows) / n}
    return rows, avg

async def run_eval():
    print("\n========== RAG 评测开始（LLM-as-judge）==========")
    rows, avg = await eval_collect()
    for x in rows:
        print(f"  · {x['q'][:20]:<22} context_precision={x['context_precision']:.2f} faithfulness={x['faithfulness']:.2f} answer_relevancy={x['answer_relevancy']:.2f}")
    print("---------- 平均指标 ----------")
    print(f"  context_precision = {avg['context_precision']:.2f}")
    print(f"  faithfulness      = {avg['faithfulness']:.2f}")
    print(f"  answer_relevancy  = {avg['answer_relevancy']:.2f}")
    print("（指标含义：context_precision=检索相关性；faithfulness=不编造；answer_relevancy=切题有用）")
    print("（生产可替换为 Ragas / AgentBench 框架，本版用 LLM-as-judge 零额外依赖、可解释）\n")

# ============================================================
# 16. 多格式摄入：格式解析器
#   说明：PDF / Word / PPT / Excel / 图片统一走 MinerU 云端（parse_document_any 分派），
#   本文件只保留本地可处理的纯文本解析，避免"两套解析路径"带来的维护分叉
# ============================================================
def parse_txt(path: str) -> list:
    # 干什么：按空行分段成块；utf-8 读入 errors=ignore 容错 gbk
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return [p.strip() for p in raw.split("\n\n") if p.strip()]

def classify(path: str) -> str:
    # 干什么：扩展名规则判断格式（快且准，绝不让 LLM 猜格式）
    # 图片格式返回 image，解析统一走 MinerU（它原生支持图片 OCR）
    ext = path.rsplit('.', 1)[-1].lower()
    return {'pdf': 'pdf', 'docx': 'docx', 'xlsx': 'excel', 'xls': 'excel',
            'pptx': 'ppt', 'txt': 'txt', 'md': 'md',
            'png': 'image', 'jpg': 'image', 'jpeg': 'image',
            'bmp': 'image', 'webp': 'image', 'gif': 'image'}.get(ext, 'unknown')

# ============================================================
# 17. 多格式摄入：MinerU 云 API 解析
#     流程：申请上传链接 -> PUT 上传 -> 轮询 -> 下载 zip 取 Markdown 按标题切块
#     注意：MinerU 不是本地包，是云端 REST API（requests 直连，Bearer token 鉴权）
# ============================================================
def parse_doc_mineru_api(path: str, model_version: str = "vlm",
                         poll_timeout: int = 600, poll_interval: int = 5) -> list:
    # 参数 path：文档绝对路径；model_version：vlm(高精度)/pipeline(快)
    # 返回：按 Markdown 标题切好的文本块列表
    if not MINERU_API_TOKEN:
        raise RuntimeError(".env 缺 MINERU_API_TOKEN（mineru.net 的 API 管理页可创建）")
    header = {"Content-Type": "application/json",
              "Authorization": f"Bearer {MINERU_API_TOKEN}"}
    # 步骤1：申请批量上传链接（返回 batch_id + 预签名上传 URL）
    body = {"files": [{"name": os.path.basename(path)}], "model_version": model_version,
            "language": "ch", "enable_formula": True, "enable_table": True}
    r = requests.post(f"{MINERU_BASE}/file-urls/batch", headers=header, json=body, timeout=30)
    r.raise_for_status()
    d = r.json().get("data", {})
    batch_id = d.get("batch_id")
    upload_url = (d.get("file_urls") or [None])[0]
    if not batch_id or not upload_url:
        raise RuntimeError(f"MinerU 申请上传链接失败: {r.text[:300]}")
    # 步骤2：PUT 上传文件二进制（官方要求此步不带 Content-Type）
    with open(path, "rb") as f:
        up = requests.put(upload_url, data=f, timeout=300)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"MinerU 上传失败: {up.status_code} {up.text[:200]}")
    # 步骤3：轮询解析状态；done 时返回体带 full_zip_url
    result_url, waited = None, 0
    while waited < poll_timeout:
        time.sleep(poll_interval); waited += poll_interval
        q = requests.get(f"{MINERU_BASE}/extract-results/batch/{batch_id}",
                         headers=header, timeout=30)
        q.raise_for_status()
        items = q.json().get("data", {}).get("extract_result", [])
        if not items:
            continue                                  # 上传还没入队，继续等
        st = items[0].get("state")
        print(f"[MinerU] state={st} ({waited}s)")
        if st == "done":
            result_url = items[0].get("full_zip_url"); break
        if st == "failed":
            raise RuntimeError(f"MinerU 解析失败: {items[0].get('err_msg')}")
    if not result_url:
        raise TimeoutError(f"MinerU 轮询超时（{poll_timeout}s），稍后重试")
    # 步骤4：下载 zip 解压到 2D_data 下的独立子目录（含 full.md + 图片目录）
    # v2：原版直接解压到 2D_data 顶层，多次导入时同名图片互相覆盖、历史产物无限堆积
    import zipfile
    ts = int(time.time())
    zip_path = os.path.join(PROCESSED_DIR, f"mineru_{ts}.zip")
    extract_dir = os.path.join(PROCESSED_DIR, f"mineru_{ts}")
    with requests.get(result_url, timeout=300, stream=True) as zr, open(zip_path, "wb") as zf:
        for c in zr.iter_content(65536):
            zf.write(c)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    os.remove(zip_path)
    # 步骤5：取本次解压出来的 Markdown，按 # 标题切块（块自带章节语义）
    mds = glob.glob(os.path.join(extract_dir, "**", "*.md"), recursive=True)
    if not mds:
        raise RuntimeError("MinerU 结果包里没有 Markdown 文件")
    full = max(mds, key=os.path.getmtime)             # 多份 md 时取最新那份
    with open(full, encoding="utf-8") as f:
        text = f.read()
    blocks = [b.strip() for b in re.split(r"\n(?=#{1,6} )", text) if b.strip()]
    return blocks or [text]

def parse_document_any(path: str, model_version: str = "vlm") -> list:
    # 干什么：统一解析入口——txt/md 本地读，其余格式（含图片）走 MinerU 云端
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "md"):
        return parse_txt(path)
    if not USE_MINERU_API:
        raise RuntimeError(f"{ext} 格式需要 MinerU 云 API：请在 .env 设 USE_MINERU_API=true")
    return parse_doc_mineru_api(path, model_version)

# ============================================================
# 18. 多格式摄入：写库 + 幂等替换 + 产物落盘
# ============================================================
def _file_key(path: str) -> str:
    # 干什么：生成文件级稳定前缀（文件名+大小 md5 前 10 位），同文件重复导入定位旧记录用
    st = os.stat(path)
    raw = f"{os.path.basename(path)}:{st.st_size}".encode()
    return "ing_" + hashlib.md5(raw).hexdigest()[:10]

def ingest_file_to_kb(path: str, target: str = "medical", model_version: str = "vlm") -> dict:
    # 干什么：摄入总入口——解析 -> 写 chroma -> 同步 BM25 注册表 -> 产物落 2D_data
    # 参数 path：文件路径或 raw_data 里的文件名；target：单库架构下固定 "medical"
    # 返回：{"source","target","chunks","status"}，status: ingested（新导入）/ replaced（同文件替换）
    # ---- 0. 路径解析：纯文件名自动拼 raw_data ----
    if not os.path.isabs(path) and not os.path.exists(path):
        cand = os.path.join(RAW_DIR, path)
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    ftype = classify(path)
    if ftype == "unknown":
        raise ValueError(f"不支持的格式：{path}")
    if target not in KB_BINDINGS:
        raise ValueError(f"target 只支持 medical（单库架构），收到 {target}")
    collection_obj, reg = KB_BINDINGS[target]
    # ---- 1. 解析成文本块 ----
    chunks = parse_document_any(path, model_version)
    if not chunks:
        raise RuntimeError("解析结果为空，请检查文件内容")
    # ---- 2. 生成稳定 id；同文件同目标重复导入 -> 删旧块实现幂等替换 ----
    # 注意：json 按"文件指纹+目标集合"区分——单库下同一文件重复导入会走幂等替换
    fkey = _file_key(path)
    ids = [f"{fkey}_{i}" for i in range(len(chunks))]
    old_json = os.path.join(PROCESSED_DIR, f"{fkey}_{target}.json")
    if os.path.exists(old_json):
        with open(old_json, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("ids"):
            collection_obj.delete(ids=old["ids"])     # 删掉 chroma 里的旧块
        # 从 BM25 语料里剔除旧块（按 id2idx 反查下标，从后往前删避免下标位移）
        old_idx = sorted((reg["id2idx"].pop(i) for i in old["ids"] if i in reg["id2idx"]), reverse=True)
        for i in old_idx:
            del reg["docs"][i], reg["metas"][i]
        # 剔除后全量下标重排：把剩余 id 的位置按从小到大排成 0..n-1（被删位置之后的整体前移）
        shift = {old_pos: new_pos for new_pos, old_pos in enumerate(sorted(reg["id2idx"].values()))}
        for k in reg["id2idx"]:
            reg["id2idx"][k] = shift[reg["id2idx"][k]]
        if old_idx:
            reg["bm25"] = BM25Okapi([tokenize(d) for d in reg["docs"]])
        status = "replaced"
    else:
        status = "ingested"
    # ---- 3. 写 chroma（余弦空间；normalize 让内积=余弦）----
    embeddings = EMBED_MODEL.encode(chunks, normalize_embeddings=True)
    metas = [{"source": os.path.basename(path)} for _ in chunks]
    collection_obj.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)
    # ---- 4. 同步 BM25 注册表（否则本轮/重启后检索都覆盖不到）----
    register_extra(reg, ids, chunks, metas)
    # ---- 5. 产物落 2D_data（重启后 register_ingested_on_startup 靠它恢复 BM25）----
    with open(old_json, "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(path), "target": target, "ftype": ftype,
                   "ids": ids, "chunks": chunks, "metadatas": metas},
                  f, ensure_ascii=False)
    print(f"[摄入] {os.path.basename(path)} -> {target} 集合 {len(chunks)} 块（{status}）")
    return {"source": os.path.basename(path), "target": target, "chunks": len(chunks), "status": status}

# ============================================================
# 19. 摄入 Agent：把"解析+入库"封装成工具，用户导入时由 agent 自主调度
#     架构：上传文件 -> agent 收到自然语言任务 -> 自动调用 ingest_document
#           -> 工具内部走 MinerU 解析 + 写向量库 + 同步 BM25 -> agent 汇报结果
# ============================================================
_INGEST_AGENT = None                                            # 摄入 agent 单例（构建一次反复用）

# ---- Cloudflare Workers AI 消息格式兼容补丁（只有走 langchain 工具调用的摄入链路会踩）----
# 实测 CF 的 OpenAI 兼容端点有两个硬要求，而 langchain 1.x 的默认序列化正好都不满足：
#   ① 助手消息带 tool_calls 时 content 必须是 ""（空字符串）——langchain 发的是 null
#      （langchain_openai 源码里是 content or None），CF 报 "Type mismatch ... 'string' not in 'null'"
#   ② 消息 content 只接受字符串——langchain 1.x 会把用户消息序列化成 block 数组
#      [{"type":"text",...}]，CF 报 "'array' not in 'string'"
# 处理：①用 monkey-patch 修序列化层；②用 middleware 把数组 content 拼回字符串（双保险）
import langchain_openai.chat_models.base as _lco_base

_cf_orig_convert = _lco_base._convert_message_to_dict           # 先保留原始函数的引用

def _cf_convert_message_to_dict(message, api="chat/completions"):
    # 干什么：包一层 langchain 的消息序列化函数，修正 CF 不认的 content 格式
    # 参数 message：langchain 的 BaseMessage；api：langchain 内部 API 标记，原样透传
    d = _cf_orig_convert(message, api)                          # 先按原逻辑转成 dict
    if d.get("role") == "assistant" and d.get("tool_calls") and d.get("content") is None:
        d["content"] = ""                                       # null -> ""，CF 实测只认空字符串
    return d

# 模块级替换：langchain_openai 内部调用点是运行时查全局名字，所以替换立即生效
_lco_base._convert_message_to_dict = _cf_convert_message_to_dict

def _cf_safe_messages(state):
    # 干什么：兜底把消息里的 block 数组 content 拼回纯字符串（CF 只收字符串）
    # 参数 state：形如 {"messages": [...]} 的状态字典；返回修正后的消息列表
    from langchain_core.messages import ToolMessage as _ToolMsg
    out = []
    for m in state["messages"]:
        if isinstance(m.content, list):
            text = "".join(b.get("text", "") for b in m.content
                           if isinstance(b, dict) and b.get("type") == "text")
            if isinstance(m, _ToolMsg):
                m = _ToolMsg(content=text, tool_call_id=m.tool_call_id)  # tool 消息必须保留 tool_call_id
            else:
                m = type(m)(content=text, id=m.id)
        out.append(m)
    return out

@tool
def ingest_document(path: str, target: str = "medical") -> str:
    """把一个已上传的文件（PDF/Word/PPT/Excel/图片/txt）解析并导入知识库：非 txt 格式自动上传 MinerU 云端高精度识别（约 1~5 分钟），识别出的文本块写入目标集合并同步 BM25 索引，完成后立刻可被检索。参数 path 为文件的完整路径（已在 raw_data 下）；参数 target 为目标集合，单库架构下固定为 medical。返回 JSON 字符串：ok（是否成功）、source（文件名）、target、chunks（块数）、status（ingested 新导入 / replaced 同文件替换）。"""
    # 干什么：给 agent 用的摄入工具——一次调用完成"MinerU 识别 -> 向量库 -> 可检索"全链路
    # 返回 JSON 字符串：agent 与后端都能可靠解析成功/失败与统计数据
    try:
        data = ingest_file_to_kb(path, target)
        data["ok"] = True
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        # 错误转 JSON 字符串返回：agent 读到后可向用户解释或换参数重试，不会中断整轮
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                          ensure_ascii=False)

def build_ingest_agent():
    # 干什么：构建摄入 agent（ReAct 循环），唯一工具是 ingest_document
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_openai import ChatOpenAI

    class CFCompatMiddleware(AgentMiddleware):
        # 干什么：langchain 1.x 的模型调用中间件——每轮调 LLM 前修正消息格式（适配 CF 端点）
        # 为什么用 middleware：create_agent 没有 langgraph 版的 prompt 函数参数，消息转换只能走钩子
        def wrap_model_call(self, request, handler):
            # 参数 request：ModelRequest，.messages 是本轮要发的全部消息
            # 参数 handler：继续执行后续流程的回调，必须返回它的结果，否则 agent 断链
            msgs = _cf_safe_messages({"messages": request.messages})   # block 数组 content 拼回字符串
            return handler(request.override(messages=msgs))            # 用修正后的消息继续原流程

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,                                          # 摄入调度要稳定，不要发散
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),           # 与主流程一致：单次调用超时快速失败
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")),
    )
    # middleware 挂上 CF 兼容层：换 DeepSeek 等标准端点时它只是原样透传，无副作用
    return create_agent(llm, tools=[ingest_document], middleware=[CFCompatMiddleware()])

def run_ingest_agent(path: str, target: str) -> dict:
    # 干什么：摄入 agent 的一次完整运行——agent 读任务 -> 自动调 ingest_document -> 汇报
    # 参数 path：已落在 raw_data 的文件完整路径；target：单库架构固定 medical
    # 返回：{"ok": 是否成功, "data": 工具返回的统计, "trace": 给前端展示的调用轨迹}
    global _INGEST_AGENT
    if _INGEST_AGENT is None:
        _INGEST_AGENT = build_ingest_agent()
    result = _INGEST_AGENT.invoke({"messages": [{"role": "user", "content":
        f"请把文件导入{target}知识库并汇报结果。文件完整路径：{path}。目标集合：{target}。"}]})
    trace = []                                                  # 调用轨迹：AI 决策/工具返回/AI 总结
    tool_payload = None                                         # 工具返回的 JSON（成功与否以此为准）
    for m in result["messages"]:
        kind = type(m).__name__
        if kind == "AIMessage" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                trace.append(f"AI 调用工具 {tc['name']} 参数={tc['args']}")
        elif kind == "ToolMessage":
            trace.append("工具返回：" + str(m.content)[:300])
            try:
                tool_payload = json.loads(m.content)
            except Exception:
                pass
        elif kind == "AIMessage":
            trace.append("AI 总结：" + str(m.content)[:300])
    ok = bool(tool_payload and tool_payload.get("ok"))
    return {"ok": ok, "data": tool_payload or {}, "trace": trace}

# ============================================================
# 20. HTTP 服务（前端页面仍从 frontend/index.html 读取，不在本文件内）
# ============================================================
def make_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi import UploadFile, File, Form
    from pydantic import BaseModel
    app = FastAPI(title="个人知识库多智能体问答系统")

    class AskReq(BaseModel):
        question: str
        mode: str = "fast"          # fast=快速通道（1 次 LLM）；full=完整多 Agent 协作 + 人工确认路径
        session_id: str = ""        # 多轮会话标识（用于槽位填充/指代消解），空则不启用会话记忆
    # 前端已拆分到 frontend/index.html，每次请求实时读文件，改前端无需重启服务
    FRONTEND_PATH = os.path.join(_BASE_DIR, "frontend", "index.html")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(FRONTEND_PATH, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    @app.post("/ask")
    async def ask(req: AskReq):
        ns = "full" if req.mode == "full" else "fast"           # 缓存按链路隔离
        # ---- 会话态：与 /ask_stream 共用同一套前置判定（追问答案不入全局缓存）----
        prep = _prepare_session(req.session_id, req.question)
        # 缓存读写与画像召回都要 encode（同步 CPU），统一 to_thread 挪出事件循环
        hit = None if prep.followup else await asyncio.to_thread(answer_cache.get, req.question, ns)
        if hit:
            # 缓存命中同样要落会话槽位（否则下一轮追问拿不到上下文线索）
            _persist_turn_session(req.session_id, prep.session, req.question, [],
                                  hit.get("query_type", ""))
            return {"answer": hit["answer"], "profile_hits": [], "event_log": ["cache"],
                    "query_type": hit.get("query_type", ""), "cached": True, "mode": ns,
                    "session": _session_payload(req.session_id, prep)}
        # v2：跳过人工确认改走 state（skip_human_check），不再切换全局 HUMAN_REVIEW（并发安全）
        ph = await asyncio.to_thread(profile_memory.recall, req.question, 3)
        res = await run_turn(req.question, ph, fast_mode=(ns == "fast"),
                             extra_state=_session_state(prep), skip_human_check=True)
        docs = res.get("docs") or []
        qt = res.get("query_type") or _guess_query_type(req.question, docs)
        sources = [m.get("source") for _, m in docs]
        _persist_turn_session(req.session_id, prep.session, req.question, docs, qt)
        # ---- 长期画像：按频率后台异步抽取（与 /ask_stream 同一逻辑）----
        profile_scheduled = False
        if req.session_id and not (qt or "").startswith(NON_RETRIEVAL_TYPES) and res.get("answer"):
            profile_scheduled = _schedule_profile_extraction(req.question, res["answer"], prep.session)
        if not prep.followup:
            await _cache_put(req.question, res.get("answer", ""), docs, qt, ns)
        return {
            "answer": res.get("answer", ""),
            "profile_hits": res.get("profile_hits", []),
            "event_log": [r["node"] for r in (res.get("event_log") or [])],
            "query_type": qt,
            "sources": sources,
            "cached": False,
            "mode": ns,
            "session": _session_payload(req.session_id, prep),
            "profile_scheduled": profile_scheduled,
        }
    @app.get("/ask_stream")
    async def ask_stream(question: str, mode: str = "fast", nocache: int = 0, session_id: str = ""):
        # 干什么：SSE 流式问答——把流程进度和回答 token 实时推给浏览器
        # 参数 question：用户问题（走 GET query 参数，方便前端用 EventSource 订阅）
        # 参数 mode：fast=快速通道（检索后一次 LLM 出答案，默认）；full=完整多 Agent 协作流程
        # 参数 nocache：1=跳过答案缓存（既不读也不写）。演示"精排开关差异"时必须用它
        # 参数 session_id：多轮会话标识——用于槽位填充/指代消解；空则不启用会话记忆
        # 事件格式：每行 "data: {json}"，type 取值 start / node / token / timing / plan / retrieval / done / error
        ns = "full" if mode == "full" else "fast"                # 缓存命名空间=链路名
        fast_mode = (ns == "fast")
        # ---- 会话态：槽位填充 / 指代判定（风险 2 的核心：检索用填充查询，生成用原问题）----
        # 与 /ask 共用同一实现；人称代词（他/她）另有类型校验，见 _fill_slots
        prep = _prepare_session(session_id, question)
        # 追问的答案依赖会话上下文，不能进全局答案缓存（否则别的会话会拿到别人的上下文答案）
        no_cache = bool(nocache) or prep.followup
        if prep.fill_note:
            print(f"[会话] {prep.fill_note} → 检索查询={prep.retrieval_query}")

        async def gen():
            nodes_seen = []                                     # 记录经过的节点名，供前端画流程条
            docs_acc = []                                       # 累积检索到的资料（拿来源）
            qt = ""                                             # 问题类型（启发式或 router 给出）
            answer_acc = ""                                     # 累积图里产出的最终答案（无流式时补推用）
            t_start = time.perf_counter()                       # 计时起点：用于统计首字/总耗时
            ttft_ms = None                                      # 首字耗时（第一个 token 到达时记录）
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}   # 本轮 checkpoint 的 thread（finally 回收）
            try:
                yield _sse({"type": "start", "mode": ns})
                # ---- 缓存优先：命中就直接分段推答案，几乎零延迟（nocache=1 时跳过） ----
                hit = None if no_cache else await asyncio.to_thread(answer_cache.get, question, ns)
                if hit:
                    for _ev in _replay(hit["answer"]):         # 40 字一段模拟流式
                        yield _ev
                    # 缓存命中也要落会话槽位（否则下一轮追问没有上下文线索）
                    _persist_turn_session(session_id, prep.session, question, [],
                                          hit.get("query_type", ""))
                    yield _sse({
                        "type": "done", "cached": True, "cache_hit": hit["hit"], "mode": ns,
                        "query_type": hit.get("query_type", ""),
                        "event_log": ["cache"], "profile_hits": [],
                        "sources": hit.get("sources") or [],
                        "ttft_ms": int((time.perf_counter() - t_start) * 1000),
                        "total_ms": int((time.perf_counter() - t_start) * 1000),
                        "session": _session_payload(session_id, prep),
                    })
                    return
                ph = await asyncio.to_thread(profile_memory.recall, question, 3)   # 长期画像召回（同 /ask，编码走线程）
                yield _sse({"type": "profile", "hits": ph})
                state_in = {
                    "query": question, "needs_retrieval": False, "query_type": "",
                    "docs": [], "analysis": "", "short_term": [],
                    "profile_hits": ph, "event_log": [], "human_feedback": "",
                    "answer": "", "fast_mode": fast_mode,        # 由 mode 参数决定链路
                    "skip_human_check": True,                    # v2：HTTP 链路不进人工确认（走 state，不再切全局开关）
                    **_session_state(prep),                      # 会话相关字段（与 /ask 同源）
                }
                # stream_mode 并存两种：custom = 节点内 writer 推的 token/计时；updates = 节点完成事件
                async for mode, payload in graph.astream(state_in, config=config,
                                                         stream_mode=["custom", "updates"]):
                    if mode == "custom":
                        if payload.get("type") == "token" and ttft_ms is None:
                            ttft_ms = int((time.perf_counter() - t_start) * 1000)   # 首个 token 到达
                        yield _sse(payload)                      # token / timing 事件原样透传
                    else:
                        for node, update in payload.items():
                            nodes_seen.append(node)
                            if isinstance(update, dict):
                                if update.get("docs"):
                                    docs_acc = update["docs"]
                                if update.get("query_type"):
                                    qt = update["query_type"]
                                if update.get("answer"):
                                    answer_acc = update["answer"]        # not_supported 等不流式的节点
                                # 召回证据事件：把"检索到了哪几块、精排有没有跑"推给前端展示
                                if update.get("retrieval_info"):
                                    yield _sse({"type": "retrieval",
                                                "node": node,
                                                "items": update["retrieval_info"],
                                                "meta": update.get("retrieval_meta") or {}})
                                # 两条链路的答案都入各自命名空间的缓存（快速/完整互不覆盖）
                                if node in ("fast_synth", "synthesizer") and update.get("answer"):
                                    if not no_cache:
                                        await _cache_put(question, update["answer"], docs_acc, qt, ns)
                            yield _sse({"type": "node", "name": node})
                # 无流式 token 的情况（如 not_supported 直接返回话术）：把最终答案补推给前端，
                # 否则页面会显示空白——这是"不调 LLM 的节点"与"流式渲染"之间的衔接点
                if ttft_ms is None and answer_acc:
                    for _ev in _replay(answer_acc):             # 无流式节点：补推静态答案
                        yield _ev
                    ttft_ms = int((time.perf_counter() - t_start) * 1000)
                total_ms = int((time.perf_counter() - t_start) * 1000)
                # ---- 回写会话槽位（供下一轮指代消解）：只写"正常检索回答"的轮次 ----
                # 澄清反问 / 能力边界话术不属于检索结果，写进去会把 domain 污染成"需要澄清"
                _persist_turn_session(session_id, prep.session, question, docs_acc, qt)
                # ---- 长期画像：按频率后台异步抽取（不阻塞响应，失败不影响主流程）----
                profile_scheduled = False
                if session_id and not (qt or "").startswith(NON_RETRIEVAL_TYPES) and answer_acc:
                    profile_scheduled = _schedule_profile_extraction(question, answer_acc, prep.session)
                yield _sse({
                    "type": "done",
                    "cached": False,
                    "mode": ns,
                    "query_type": qt or _guess_query_type(question, docs_acc),
                    "event_log": nodes_seen,
                    "profile_hits": ph,
                    "sources": [m.get("source") for _, m in docs_acc],
                    "ttft_ms": ttft_ms if ttft_ms is not None else total_ms,
                    "total_ms": total_ms,
                    # 会话与检索改写的可观测信息（前端展示，便于验证指代消解是否生效）
                    "session": _session_payload(session_id, prep),
                    "profile_scheduled": profile_scheduled,
                })
            except Exception as e:
                # 异常也走 SSE 返回，前端能显示错误而不是静默断流
                yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                # v2：移除 finally 里的 HUMAN_REVIEW 恢复——本轮是否跳过确认已改由 state.skip_human_check 表达
                # 本轮 checkpoint 用完即删（含 docs 原文）：MemorySaver 只写不回收，长跑会让内存单调增长
                _drop_thread(config)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no",     # 关掉代理缓冲，保证逐字到达
                                          "Connection": "keep-alive"})
    @app.get("/profile")
    async def get_profile():
        return {"profiles": profile_memory.dump_all()}
    @app.get("/audit")
    async def get_audit():
        return {"events": event_logger.records[-20:]}
    @app.post("/eval")
    async def post_eval():
        # v2：eval_collect 内部已逐题 skip_human_check=True，不再切换全局 HUMAN_REVIEW
        rows, avg = await eval_collect()
        return {"rows": rows, "avg": avg}
    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...), target: str = Form("medical")):
        # 干什么：多格式知识上传入口（PDF/Word/PPT/Excel/图片/txt），agent 驱动版
        # 流程：原始文件落 raw_data/ -> 摄入 agent 收到自然语言任务 -> agent 自主调用
        #       ingest_document 工具（内部走 MinerU 解析 + 写 chroma + 同步 BM25）
        #       -> 返回 agent 调用轨迹与统计
        # 参数 target：单库架构固定 medical（前端不再传，保留参数兼容旧调用）
        if target not in KB_BINDINGS:
            return {"ok": False, "error": "target 只支持 medical（单库架构）"}
        # v2：basename 剥掉路径分隔符——原版直接用 file.filename，传 "../../x.pdf" 会写出 raw_data 之外
        save_path = os.path.join(RAW_DIR, os.path.basename(file.filename))
        with open(save_path, "wb") as f:           # 原始文件先落 raw_data，符合"原始/产物分离"规则
            f.write(await file.read())
        try:
            # agent.invoke 是同步阻塞调用（LLM 循环+可能数分钟云端解析），
            # asyncio.to_thread 把它丢进线程池执行，避免卡死 FastAPI 的事件循环
            result = await asyncio.to_thread(run_ingest_agent, save_path, target)
            out = {"ok": result["ok"], "trace": result["trace"]}
            out.update(result.get("data") or {})                # source/target/chunks/status 摊平到顶层
            return out
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return app

# ============================================================
# 21. 主入口：serve 起服务，否则进交互问答
# ============================================================
if __name__ == "__main__":
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("your_"):
        raise SystemExit("请先在 .env 里把 OPENAI_API_KEY 换成真实的 API Token（Cloudflare Workers AI / DeepSeek 等）")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    model = os.getenv("OPENAI_MODEL") or "gpt-4o"
    # v2：补齐超时与重试配置（原来这里是默认 600s + 无限重试，与主链路的 LLM_TIMEOUT 不一致）
    _cli_client = AsyncOpenAI(api_key=key, base_url=base_url,
                              timeout=float(os.getenv("LLM_TIMEOUT", "30")),
                              max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")))
    set_default_openai_client(_cli_client)
    set_default_openai_api("chat_completions")
    print(f"已配置：端点={base_url or 'OpenAI官方'}  模型={model}\n")
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        import webbrowser
        import threading
        def _open_browser():
            time.sleep(2)
            webbrowser.open("http://127.0.0.1:8008")            # 服务端口 8008
        threading.Thread(target=_open_browser, daemon=True).start()
        print("启动 FastAPI 服务：http://127.0.0.1:8008")
        uvicorn.run(make_app(), host="127.0.0.1", port=8008)    # 端口 8008
    else:
        # ==================== 交互问答模式（命令行 REPL）====================
        MAX_SHORT_TURNS = 5                                     # 短期记忆最多保留 5 轮（每轮 user+assistant 两条消息）
        async def main():
            # 干什么：命令行问答主循环——读问题 -> 跑图 -> 打印答案，附带三个本地管理命令
            # 与 HTTP 端共用同一张图与同一套会话逻辑，只是没有 SSE 流式与会话槽位（按单会话处理）
            print(f"知识库就绪：{len(MEDICAL_CHUNKS)} 个块（索引已在模块加载时构建）\n")
            print("自由提问模式（exit/quit/退出 结束；/profile 看画像；/audit 看决策链；/eval 跑评测）\n")
            short_term_history = []                             # 短期对话历史（user/assistant 消息对），跨轮滚动保留
            turn_count = 0                                      # 画像抽取计数器（与 HTTP 端统一按 PROFILE_EVERY 频率）
            while True:
                try:
                    q = input("你的问题> ").strip()
                except EOFError:
                    break                                       # 管道/脚本喂入结束时退出（无 Ctrl+D 交互场景）
                if not q:
                    continue                                    # 空回车不当作提问
                if q.lower() in ("exit", "quit", "退出"):
                    print("已退出。")
                    break
                if q in ("/profile", "画像"):                    # 本地命令①：查看长期画像库全部内容
                    all_p = profile_memory.dump_all()
                    print("\n【长期记忆库中的用户画像】")
                    print("\n".join(f"  - {p}" for p in all_p) if all_p else "  (空，多聊几轮后会自动积累)")
                    print("-" * 60)
                    continue
                if q in ("/audit", "审计"):                      # 本地命令②：查看决策链审计日志（最近 10 条）
                    print("\n【可追溯决策链（最近 10 条）】")
                    for r in event_logger.records[-10:]:
                        print(f"  [{r['ts']}] {r['node']}: in={r['in']} | out={r['out']}")
                    print("-" * 60)
                    continue
                if q in ("/eval", "评测"):                       # 本地命令③：跑一次 LLM-as-judge 评测
                    await run_eval()
                    continue
                print()
                # 每轮都先召回相关长期画像（内部有 PROFILE_MIN_SIM 阈值过滤，无关问题召回空）
                profile_hits = await asyncio.to_thread(profile_memory.recall, q, 3)
                if profile_hits:
                    print(f"[长期记忆] 召回 {len(profile_hits)} 条相关画像：{profile_hits}")
                # 跑完整流程；交互模式不传 fast_mode → 默认走 router→analyzer→人工确认→synthesizer 三段式
                result = await run_turn(q, profile_hits, short_term_history)
                # 用图返回的 short_term 覆盖本地历史（图内已按 add_messages 追加语义合并）
                short_term_history = result.get("short_term") or short_term_history
                # 超过 5 轮（10 条消息）就从头裁掉旧消息，控制上下文长度
                if len(short_term_history) > MAX_SHORT_TURNS * 2:
                    short_term_history = short_term_history[-MAX_SHORT_TURNS * 2:]
                print("【回答】")
                print(result["answer"])
                print("-" * 60)
                # v2：交互模式的画像抽取与 HTTP 端统一——同样按 PROFILE_EVERY 频率、同样走 _filter_profile_points
                # （原版每轮都抽且不过滤，模型输出"未透露/不抽取…"这类解释文字会被当成画像入库）
                turn_count += 1
                if PROFILE_EVERY > 0 and turn_count % PROFILE_EVERY == 0:
                    # 到达抽取频率：调画像抽取 Agent，从本轮问答里挖"跨会话稳定的偏好/背景"
                    ex_res = await Runner.run(profile_extractor_agent,
                                              f"用户问题：{q}\n回答：{result['answer']}")
                    new_points = _filter_profile_points(str(ex_res.final_output).split("\n"))
                    if new_points:
                        await asyncio.to_thread(profile_memory.add, new_points)   # 内部自带相似度去重（<0.15 跳过）
        asyncio.run(main())                                     # 异步入口：图节点全是 async，REPL 也跑在事件循环里
