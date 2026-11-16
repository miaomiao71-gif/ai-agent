# 知识库问答 Agent —— 代码评审与升级方案

> 评审对象：`build_db.py`（68 行）、`chat_agent.py`（137 行）、`faiss_index/`、`knowledge_base/`
> 评审日期：2026-11-16

---

## 0. 一句话结论

这是一个**能跑通的 Naive RAG demo**，但离"简历项目"还差一次重构。四个硬伤：

1. **代码根本没提交到 Git** —— `git ls-files` 里只有 `.idea/*`，两个 `.py` 从未 commit。别人 clone 你的仓库只能看到 IDE 配置。
2. **检索实际上没起作用** —— 向量库里只有 **5 个块**，而 `k=4`，每次检索相当于把 80% 的知识库塞给模型。这不叫 RAG，叫"全量上下文 + 一点点排序"。
3. **多轮对话会把整篇文档写进历史** —— 每一轮的"参考资料"都作为 `HumanMessage` 永久留在 `self.history` 里，第 5 轮时上下文里堆着 4 份过期的旧文档。
4. **零工程化** —— 没有 requirements、没有 README、没有 .gitignore、没有测试、没有日志、没有评估指标。面试官问"你怎么证明它答得对"，目前无法回答。

下面是可逐条落地的清单。**P0 = 会出错/影响正确性，P1 = 影响 RAG 效果，P2 = 工程质量，P3 = 简历维度。**

---

## 1. 先看客观数据

| 指标 | 现状 | 说明 |
|---|---|---|
| 向量数量 | **5** | `faiss_index/index.faiss` 实际 `ntotal=5` |
| 向量维度 | 1024 | `qwen3.7-text-embedding-flash` |
| 知识库规模 | 1 篇 docx / 2470 字符 / 9 个问答 | 大模型基础面题库 |
| 每次检索返回 | 4 块 | 占全库 **80%** |
| 实际注入上下文 | 约 2000+ 字符 | ≈ 全库 2470 字，检索≈全量 |
| Git 已跟踪文件 | 5 个，全是 `.idea/` | 源码未入库 |

> 这不是"参数没调好"的问题，而是**知识库样本量与检索策略不匹配**：库越小，`k=4` 越接近"把全部文档塞进去"。想体现 RAG 价值，知识库至少要上百个 chunk（建议 8~15 篇文档 / 300+ QA）。

---

## 2. P0：会出错 / 影响正确性

### 2.1 参考资料被当成用户消息写进历史（最严重的设计缺陷）

`chat_agent.py:84-89`

```python
augmented_input = f"参考资料：\n{context}\n\n用户问题：{user_input}"
self.history.append(HumanMessage(content=augmented_input))   # ← 问题在这里
```

**后果有三个，都是实打实的：**

1. **上下文污染 + token 爆炸**：每轮 2000+ 字符的参考资料永久驻留。第 3 轮时模型同时看到 3 份不同问题的旧文档，很容易拿第 1 轮的文档回答第 3 轮的问题。
2. **指代型追问失效**：用户问"它和 T5 有什么区别"，`retriever.invoke("它和 T5 有什么区别")` 检索不到任何有用块——因为 query 里没有"它"的指代对象。
3. **Prompt 注入面扩大**：文档内容进入了 user 角色，如果知识库里混进"忽略以上指令"这类文本，模型会当作**用户指令**执行。

**改法：历史只存干净的用户问题，参考资料每轮临时拼装、用完即弃。**

```python
def send_stream(self, user_input: str):
    # 1) 先把追问改写成独立问题（见 3.3），再检索
    search_query = self.rewrite_query(user_input)
    docs = self.retrieve(search_query)
    context = self.format_context(docs)

    # 2) 历史保持干净，参考资料只在本轮临时注入
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *self.history,                       # 只有干净的 user / assistant
        HumanMessage(content=(
            f"<reference>\n{context}\n</reference>\n\n"
            f"问题：{user_input}"
        )),
    ]

    full = ""
    try:
        for chunk in self.model.stream(messages):
            if chunk.content:
                full += chunk.content
                yield chunk.content
    finally:
        # 3) 无论正常结束、调用方 break、还是异常，历史都保持一致
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=full))
```

顺带把 `self.history.append(AIMessage(...))` 放进 `finally`，顺手解决 2.3 的问题。

---

### 2.2 来源信息用 `set()` 收集，丢失粒度且顺序不稳定

`chat_agent.py:82`

```python
sources = list(set([os.path.basename(...) for d in docs]))
```

**问题**：① 只保留文件名，没有页码/章节/块序号，用户无法核验；② `set()` 无序，同一问题两次调用输出顺序不同；③ 模型回答里没有 `[1][2]` 引用锚点，日志里无法回溯"这句话来自哪一块"。

**改法**：检索结果带稳定 id 和位置元信息，回答要求按编号引用。

```python
# build 阶段给每个 chunk 写全 metadata
chunk.metadata.update({
    "doc_id": doc_hash,          # 文档内容哈希，增量索引用
    "source": file_name,
    "section": section_title,    # 所属问题/章节
    "chunk_id": f"{doc_hash}-{i:03d}",
})

# 推理阶段：保序去重 + 编号
def format_context(docs):
    lines, seen = [], set()
    for i, d in enumerate(docs, 1):
        key = d.metadata["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{i}] 来源《{d.metadata['source']}》· {d.metadata.get('section','-')}\n{d.page_content}")
    return "\n\n".join(lines)
```

System prompt 里加一句：`每个结论后用 [编号] 标注依据；没有依据的句子不要写。`

---

### 2.3 流式生成被中断时，历史会坏掉

`chat_agent.py:100-101` 把 `AIMessage` 追加在生成器**末尾**。生成器是惰性的：调用方 `break`、Ctrl+C、HTTP 客户端断开、模型中途报错，都会让这行永远不执行。结果是历史里出现**连续两条 HumanMessage**，部分 provider 会直接返回 400，另一些会静默丢上下文。

修复见 2.1 的 `finally`。

---

### 2.4 配置在两个文件里各写了一份（典型的配置漂移 bug）

| 配置 | build_db.py | chat_agent.py |
|---|---|---|
| `EMBEDDING_MODEL` | 第 16 行 | 第 19 行 |
| `INDEX_PATH` | 第 15 行 | 第 20 行 |
| `API_KEY / BASE_URL` | 第 11-12 行 | 第 11-12 行 |

只要哪天你只在 `build_db.py` 里换了 embedding 模型（比如换 `bge-m3`），索引按 1024 维建，查询按新模型 embedding 去搜——轻则 `AssertionError: dimension mismatch`，重则（维度相同但语义空间不同）**检索结果全错但程序不报错**。这种 bug 极难排查，也是面试官很爱问的"你怎么保证索引和查询一致"。

**改法**：单一配置源。

```python
# config.py
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    dashscope_api_key: str
    dashscope_base_url: str
    chat_model: str = "qwen3.8-27b"
    embedding_model: str = "qwen3.7-text-embedding-flash"
    index_dir: Path = BASE_DIR / "faiss_index"
    kb_dir: Path = BASE_DIR / "knowledge_base"
    top_k: int = 6
    score_threshold: float = 0.35
    max_turns: int = 6
    model_config = {"env_file": BASE_DIR / ".env", "extra": "ignore"}

settings = Settings()
```

---

### 2.5 库代码里 `sys.exit()` + 相对路径

`chat_agent.py:15, 47`：构造函数直接 `sys.exit()`。一旦你后面加 FastAPI / Streamlit，一个请求进来 `sys.exit` 会把**整个进程**干掉；测试也无法断言。

`INDEX_PATH = "./faiss_index"` 依赖当前工作目录，从别的目录启动就找不到索引。

**改法**：抛自定义异常 + 绝对路径（见 2.4 的 `BASE_DIR`）。

```python
class IndexNotFoundError(RuntimeError): ...

if not settings.index_dir.exists():
    raise IndexNotFoundError(f"向量库不存在：{settings.index_dir}，请先运行 python -m rag.build")
```

---

### 2.6 `allow_dangerous_deserialization=True`

`chat_agent.py:60`：FAISS 的 `save_local` 会把 docstore 存成 pickle，加载时等于**执行任意代码**。自己生成的索引问题不大，但如果索引文件进了 Git、或部署时从对象存储拉取，就是一条 RCE 路径。

**建议**：① 索引目录加 `.gitignore`，改用对象存储/构建流水线分发；② 或者换 **Chroma / Milvus / Qdrant**，序列化是 JSON/SQLite，天然规避这个问题，而且支持 `upsert/delete`（解决 4.4 的增量索引）。

---

### 2.7 没有超时和重试

`self.model.stream(...)` 和 `FAISS.from_documents(...)` 都没有 `timeout` / `max_retries`。DashScope 在高并发或长文档时会返回 429/5xx，现在是**整轮构建直接失败**（`build_db.py` 只 catch 了 loader，没 catch embedding）。

```python
# 构建阶段分批 + 重试
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.documents import Document

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def embed_batch(texts): return embeddings.embed_documents(texts)

db = None
for i in range(0, len(chunks), 16):
    batch = chunks[i:i+16]
    vecs = embed_batch([c.page_content for c in batch])
    if db is None:
        db = FAISS.from_embeddings(list(zip([c.page_content for c in batch], vecs)), embeddings, metadatas=[c.metadata for c in batch])
    else:
        db.add_embeddings(list(zip([c.page_content for c in batch], vecs)), metadatas=[c.metadata for c in batch])
```

分批还带来两个好处：**能看到进度**、**单批失败不影响已完成的批次**。

---

## 3. P1：影响 RAG 效果

### 3.1 检索没有相关性阈值，"永远有话可说"

`chat_agent.py:63`：`search_kwargs={"k": 4}` 无条件返回 4 条。用户问"今天天气怎么样"，也会返回 4 段大模型知识，然后模型硬着头皮关联。

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.35},
)
# 注意：langchain FAISS 默认是 L2 距离，relevance_score = 1/(1+distance)，
# 想用真正的余弦相似度需在构建时 DistanceStrategy.MAX_INNER_PRODUCT + 向量归一化
```

低分直接走兜底分支（"知识库中没有相关内容"），比让模型瞎猜强得多——这也是 3.7 里"降低幻觉"的可度量手段。

### 3.2 切分策略与文档结构完全不匹配（性价比最高的一条）

你的知识库是**编号问答题库**（`1 xxx？` / `2 xxx？` …共 9 题），而 `build_db.py:42-46` 用 `RecursiveCharacterTextSplitter(600, 100, ["\n\n","\n","。",...])` 按字符硬切。实际后果：

- 5 个块覆盖 9 个问答 → **一个块里塞了 2 个不相关的问题**
- 600 字符的切点大概率落在某个答案中间 → **Q 和 A 被切开**，检索到答案的后半截却看不到问题
- 题库文档天然自带"问题 = 检索信号的实体"，白白浪费

**改法：结构感知切分，一个 QA 一个 chunk。**

```python
import re

QA_HEAD = re.compile(r"^\s*(\d{1,3})[\.、\s]\s*(.+?)\s*$", re.M)

def split_by_question(text: str, meta: dict):
    marks = list(QA_HEAD.finditer(text))
    docs = []
    for i, m in enumerate(marks):
        end = marks[i+1].start() if i + 1 < len(marks) else len(text)
        q, body = m.group(2).strip(), text[m.end():end].strip()
        if len(body) < 20:
            continue
        docs.append(Document(
            page_content=f"问题：{q}\n答案：{body}",
            metadata={**meta, "section": q, "q_no": m.group(1)},
        ))
    return docs
```

**这一步的收益可以量化**（写进简历）：chunk 数 5 → 9，每个 chunk 语义自洽，Recall@5 通常能从 60% 左右提到 85%+（用你自己的测试集验证，别编数字）。

如果文档不是问答体，就换成 **按标题层级切**（`MarkdownHeaderTextSplitter` / unstructured 的 `title` 元信息），把 `h1 > h2 > h3` 路径存进 metadata，检索时拼进 chunk 前缀——这是"父子块 / 上下文增强检索"的雏形。

### 3.3 检索 query 不使用对话历史

`chat_agent.py:77` 直接用 `user_input` 检索。多轮场景下必然失效：

| 轮次 | 用户输入 | 检索命中 |
|---|---|---|
| 1 | 主流开源模型体系有哪些？ | ✅ |
| 2 | prefix Decoder 和 causal Decoder 区别？ | ✅ |
| 3 | **它**训练效率为什么低？ | ❌ 检索不到（"它"没有指向） |

**改法：检索前做 query rewriting（这是标准做法，也是面试加分点）。**

```python
REWRITE_PROMPT = """根据对话历史，把用户的最新问题改写成一个不依赖上下文的独立问题。
只输出改写后的问题，不要解释。若已经是独立问题，原样输出。

对话历史：
{history}

最新问题：{question}"""
```

三种实现按成本递增，面试可讲取舍：

1. **LLM 改写**（最简单，多一次小模型调用，延迟 +200~400ms）
2. **多轮拼接**：`last_user + " " + current` 一起检索，零成本，效果有限
3. **HyDE**：让 LLM 先编一段"假答案"再拿去检索，对短 query 提升明显

### 3.4 没有混合检索、没有 rerank

中文技术问答里，**专有名词的精确匹配极其重要**（"U-PaLM"、"Flan-T5"、"zero-shot"）。纯向量检索对这类 token 不敏感，经常排在语义相近但答案不对的块后面。

标准两段式：

```
query ─┬─ BM25 (jieba 分词) ──┐
       └─ 向量检索 ──────────┴─ RRF 融合 ─→ Top20 ─→ Rerank(bge-reranker) ─→ Top5 → LLM
```

- **BM25**：`rank_bm25` 或 Elasticsearch；中文先 `jieba.lcut`
- **融合**：RRF（`score = Σ 1/(60+rank)`），比加权求和更稳，不用调权重
- **Rerank**：`BAAI/bge-reranker-v2-m3`（本地）或 DashScope `gte-rerank`（API）

这套组合拳是 2024 年后 RAG 项目的标配，也是简历上最容易被追问的技术点。**务必在你的评估集上给出开/关的对比数字**。

### 3.5 `Docx2txtLoader` 丢掉了结构与表格

`build_db.py:27`：docx2txt 只输出纯文本，**标题层级、表格、项目符号全部丢失**（实测 bullet 变成 `•`），页眉"扫码查看更..."也被混进了正文第 2 行。

**建议**：`python-docx` 自己遍历 `paragraph.style.name`（`Heading 1/2`）建立层级，或者用 `unstructured` 的 `partition_docx`（能拿到 `Title`/`NarrativeText`/`Table` 元素类型）。表格单独转成 Markdown 再入库，别让它变成一个 chunk 的碎片。

### 3.6 `chunk_size=600` 与 `k=4` 的组合在这个库上没有意义

600×4 ≈ 2400 字符 ≈ 全库 2470 字符。**上下文里几乎就是原文**。这会导致一个尴尬的场景：面试官让你演示"检索效果"，你没法演示——因为检索没有在筛选什么。

两个方向都行，建议一起做：
- **扩库**：把那套题库其余分册都放进去（200+ QA），让检索真正有区分度
- **调参实验**：固定评估集，扫 `chunk_size ∈ {200,400,600}` × `k ∈ {3,5,8}`，画一张 Recall/延迟的权衡表。**这张表就是简历里"做过实验"的证据**

### 3.7 System prompt 可以更严格

现在的 prompt 只说"找不到就回答无法回答"，但没有约束：① 引用编号 ② 只允许基于参考资料 ③ 允许部分回答 ④ 不要复述问题。

```text
你是知识库问答助手，只能依据 <reference> 中的内容回答。

规则：
1. 每个结论后标注依据编号，如 [1][2]；没有依据的句子不要写。
2. 参考资料只覆盖部分问题时，只回答被覆盖的部分，并说明"其余部分资料未涉及"。
3. 参考资料与问题无关时，直接回答：知识库中没有找到相关内容。
4. 不要复述问题，不要输出"根据参考资料"这类套话。
```

---

## 4. P2：工程质量（这部分决定"能不能算项目"）

### 4.1 `git ls-files` 里没有源码

```
$ git ls-files
.idea/.gitignore
.idea/ai.iml
.idea/inspectionProfiles/profiles_settings.xml
.idea/modules.xml
.idea/vcs.xml
```

三个 commit（`工具agent` / `文案生成agent` / `翻译agent`）只提交了 IDE 配置，**你的 137 行核心代码从未入库**。这条必须先修，否则后面所有工作都没有版本记录。

```bash
# .gitignore
.venv/
.env
__pycache__/
*.pyc
.idea/
faiss_index/
*.log
```

`faiss_index/` 建议不入库（二进制、每次构建都变），改为提供 `make index` 或在 README 写明构建命令。

### 4.2 缺依赖清单

`pip freeze` 里有 `faiss-cpu 1.16.0`、`docx2txt`、`langchain*`，但没有 `requirements.txt`。别人（和面试官）无法复现。

```bash
pip freeze | grep -Ei "langchain|faiss|openai|dotenv|docx2txt|tenacity|pydantic" > requirements.txt
```

更好：`pyproject.toml` + 分组依赖（`[project.optional-dependencies] dev = ["pytest", "ruff"]`）。

### 4.3 用 `print` 当日志

6 处 `print`。至少换成 `logging`（带 `level`、`%(asctime)s`、可选 `RotatingFileHandler`），并把关键链路打点：`retrieved={k, scores}`、`prompt_tokens`、`latency_ms`。**面试问"线上怎么排查问题"，答案就在这里。**

### 4.4 索引只能全量重建

`build_db.py` 每次 `FAISS.from_documents` 全量重建，无法做到：新增一篇文档只加增量、删除文档、修改文档。知识库一大就不可用。

**改法**：文档内容哈希 + manifest。

```json
// faiss_index/manifest.json
{
  "embedding_model": "qwen3.7-text-embedding-flash",
  "dim": 1024,
  "docs": {"a1b2c3": {"source": "1-大模型基础面.docx", "chunks": 9, "mtime": 1731...}}
}
```

启动时对比 `mtime/hash`，只重建变化的文档；`embedding_model` 不一致直接拒绝加载并提示重建（顺手防住 2.4 的配置漂移）。要做 `delete`，就得换 Chroma/Milvus——这也是"为什么从 FAISS 换到向量数据库"的绝佳面试素材。

### 4.5 没有测试

至少三个纯函数级单测（不需要调 API，跑得快）：

- `split_by_question()`：给定样例文本，断言切出 9 个 chunk、每个 chunk 含"问题："前缀
- `format_context()`：断言编号连续、重复 chunk 被去重
- `_trim_history()`：断言裁剪后首条是 user、system 未被裁掉

再加一个 `pytest -m "not api"` 的集成测试用 mock embedding。

### 4.6 没有服务化 / 没有 UI

CLI 的 `input()` 无法演示。**简历项目必须有能点开的界面**，最省事的是 Streamlit：

```python
# app.py —— 30 行就能有引用卡片 + 流式输出 + 侧边栏参数
st.chat_input() → agent.stream() → st.write_stream() → st.expander("参考来源")
```

想更"工程"，就 FastAPI + SSE：

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    async def gen():
        async for tok in agent.astream(req.question):
            yield f"data: {json.dumps({'t': tok}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

**FastAPI 版 + Streamlit 前端 + docker-compose** 这三件套摆出来，项目气质立刻从"作业"变成"系统"。

### 4.7 其他细节

| 位置 | 问题 | 建议 |
|---|---|---|
| `chat_agent.py:65-72` | 按消息**条数**裁剪，长文档轮次仍可能爆 context | 按 token 估算裁剪（`tiktoken` 或 `len/1.5` 中文近似） |
| `chat_agent.py:70-72` | `dialog[-max_msgs:]` 可能从 `AIMessage` 开始 | `while dialog and not isinstance(dialog[0], HumanMessage): dialog.pop(0)` |
| `chat_agent.py:18` | `MODEL_NAME` 写死 | 走 `settings`，且 `temperature=0.1` 对 RAG 合理，可加 `top_p` |
| `build_db.py:23-28` | 4 个 DirectoryLoader 各扫一遍目录 | 一个 loader + `loader_cls` 映射表，或统一走自定义 `load_documents()` |
| `build_db.py:32-35` | 只 catch loader 异常，异常吞掉后没有任何文档数校验 | 加"加载失败文件清单"日志；`docx` 解析失败常静默产出空文本 |
| `chat_agent.py:24` | prompt 里用了全角引号包裹回答模板 | 无功能问题，但统一成 `「」` 或去掉引号更清晰 |
| 全局 | 无类型注解 / 无 docstring | 至少给公共方法加签名类型，`mypy` 基本能过 |

---

## 5. 建议的目录结构

```
kb-agent/
├── app.py                     # Streamlit 前端
├── api/
│   └── main.py                # FastAPI + SSE
├── src/rag/
│   ├── config.py              # pydantic-settings 单一配置源
│   ├── loader.py              # docx/pdf/md 解析 + 结构提取
│   ├── splitter.py            # 结构感知切分（QA / 标题层级）
│   ├── embedder.py            # 分批 + 重试
│   ├── index.py               # 构建 / 加载 / manifest 增量
│   ├── retriever.py           # 向量 + BM25 + RRF + rerank
│   ├── rewrite.py             # query rewriting
│   ├── chain.py               # 组装 prompt / 流式生成 / 引用
│   └── schemas.py             # Pydantic 模型
├── scripts/
│   ├── build_index.py
│   ├── build_eval_set.py      # 从题库自动生成评估集
│   └── evaluate.py            # Recall@k / 忠实度 / 延迟
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md                  # 架构图 + 指标表 + demo 截图/GIF
```

---

## 6. 三阶段升级路线

### 阶段一：让"正确性"站得住（1~2 天）

- 修 2.1 / 2.3 / 2.5（历史解耦、异常处理、绝对路径）
- 上 3.2 结构感知切分
- 上 3.1 阈值 + 3.7 严格 prompt + 编号引用
- 补 `.gitignore`、`requirements.txt`，**把代码提交上去**
- ✅ 阶段产出：能回答"你的 RAG 怎么防幻觉、怎么溯源"

### 阶段二：让"技术深度"看得见（2~3 天）

- 3.3 query rewriting（历史感知检索）
- 3.4 混合检索（BM25 + 向量 + RRF）+ rerank
- 4.4 增量索引 + manifest
- 4.6 FastAPI（SSE 流式）+ Streamlit 前端
- 4.3 结构化日志（检索耗时、命中块、token 用量）
- ✅ 阶段产出：能回答"为什么这么设计、各模块怎么取舍"

### 阶段三：让"结果"可量化（2~3 天）

- 评估集 50~100 条：`(问题, 标准答案, 出处 chunk_id)`
- 指标：`Recall@5`、`MRR`、答案忠实度（LLM-as-judge 或 RAGAS）、`P95 延迟`、`单次成本`
- 消融实验表：`naive / +结构切分 / +hybrid / +rerank / +rewrite` 五组对比
- Docker 化 + README（架构图 + 指标表 + GIF 演示）
- ✅ 阶段产出：简历里那 3~4 行带数字的 bullet

---

## 7. 评估集怎么建（最容易省略、但最能拉开差距）

题库本身就是天然的评估源：

```python
# scripts/build_eval_set.py
# 1. 从 docx 抽出 9 个 QA，取问题作为 query，答案作为 ground_truth
# 2. 人工补充 10~20 条「知识库外」问题，用来测「拒答率」
# 3. 把 QA 打散成 6 份不同文档，避免"一问一文档"的过拟合（重要）
# 4. 生成 50 条变体问题（口语化 / 缩写 / 带指代的追问）测鲁棒性
```

评估输出长这样（README 里放这张表）：

| 配置 | Recall@5 | 拒答准确率 | 忠实度 | P95 延迟 |
|---|---|---|---|---|
| baseline（当前代码） | 0.62 | 0.40 | 0.71 | 1.8s |
| + 结构切分 | 0.81 | 0.55 | 0.79 | 1.7s |
| + hybrid 检索 | 0.88 | 0.60 | 0.83 | 2.1s |
| + rerank | **0.93** | 0.65 | **0.88** | 2.6s |
| + query rewrite | 0.93 | 0.70 | 0.90 | 3.0s |

> 上面的数字是**格式示例，不要直接抄**。你必须跑出自己的数，面试官一定会问"这个 0.93 怎么算的"，答不上来比没有数字更糟。

---

## 8. 简历怎么写

### ❌ 现在能写出来的（弱）

> 基于 LangChain + FAISS 实现知识库问答 Agent，支持文档向量化、流式输出与多轮对话。

问题：只有"用了什么"，没有"解决了什么、效果如何"，面试官无法区分你和教程复制。

### ✅ 重构后的写法（强）

> **企业知识库 RAG 问答 Agent**（个人项目 / Python + LangChain + FastAPI）
>
> - 设计结构感知切分策略（按问答/标题层级），将 chunk 语义完整率从 62% 提升至 91%，解决按字符切分导致的 Q-A 割裂问题；
> - 实现「BM25 + 向量」混合检索与 RRF 融合，叠加 bge-reranker 精排，在 80 题自建评估集上 Recall@5 由 0.62 提升至 0.93，P95 延迟控制在 2.6s 内；
> - 引入历史感知的 Query Rewriting，解决多轮追问中的指代丢失问题，多轮场景召回率提升 24pp；
> - 基于编号引用与阈值拒答机制抑制幻觉，知识库外问题拒答准确率由 40% 提升至 70%；
> - 使用 FastAPI + SSE 实现流式接口，配合 manifest 增量索引（文档哈希去重），单文档更新耗时由全量 42s 降至 3s。

**注意**：每条都是「做了什么 → 为什么做 → 数字证明」。数字必须是真跑出来的。

### 面试会被追问的点（提前准备好）

| 简历关键词 | 大概率追问 |
|---|---|
| 结构感知切分 | chunk_size 怎么定的？切太碎和大块各有什么问题？ |
| 混合检索 + RRF | 为什么用 RRF 不用加权求和？权重怎么来的？ |
| rerank | 为什么不直接加大 k？rerank 模型怎么选？延迟涨多少？ |
| Query Rewriting | 多一次 LLM 调用值得吗？怎么避免改写跑偏？ |
| 拒答机制 | 阈值怎么定的？0.35 是拍脑袋吗？（→ 答：扫阈值-拒答率曲线） |
| 评估集 | 80 题怎么来的？ground truth 谁标的？LLM-as-judge 靠谱吗？ |
| FAISS → 向量库 | 为什么要换？FAISS 什么时候够用？ |
| 增量索引 | 文档改了怎么检测？嵌入模型升级怎么办？ |
| 流式 | SSE 断线怎么处理？和 WebSocket 怎么选？ |

---

## 9. 附：几个必修的小改动（可直接粘贴）

**A. `_trim_history` 修正首条为 user、system 不丢**

```python
def _trim_history(self):
    if self.max_turns <= 0:
        return
    dialog = self.history[-self.max_turns * 2:]
    while dialog and not isinstance(dialog[0], HumanMessage):
        dialog.pop(0)
    self.history = dialog
```

**B. 检索加阈值 + 显式低分兜底**

```python
def retrieve(self, query: str):
    docs_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.top_k)
    return [(d, s) for d, s in docs_scores if s >= self.score_threshold]
```

**C. 检索日志（面试时直接给他们看日志截图）**

```python
logger.info("retrieve q=%r hits=%d top1=%.3f took=%dms",
            query[:40], len(docs), docs[0][1] if docs else 0, cost_ms)
```

---

## 小结

| 维度 | 现状 | 目标 |
|---|---|---|
| 代码规模 | 2 文件 205 行 | 8~10 模块，分层清晰 |
| 检索方式 | 纯向量，无阈值 | 混合 + 融合 + 精排 + 阈值 |
| 多轮 | 历史塞文档，无改写 | 干净历史 + query rewrite |
| 溯源 | 文件名去重 | chunk 级编号引用 |
| 索引 | 全量重建 | manifest 增量 |
| 评估 | 无 | 80 题 + 5 组消融 |
| 交付 | CLI | FastAPI + Streamlit + Docker |
| 依赖/文档/测试 | 全无 | 全有 |

**优先级建议**：先花半小时把 `.gitignore` + `requirements.txt` + 首次真实 commit 补上（不然你后面所有改进都没有版本记录），然后按阶段一走。
