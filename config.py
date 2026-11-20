"""统一配置管理。

所有可调参数集中在此，可用 .env 文件或环境变量覆盖，避免参数散落在各个脚本里。
字段名大小写不敏感；同时兼容旧版 .env 中的 MODEL_NAME 写法。

面试讲解要点：
- 用 pydantic-settings 的 BaseSettings 做类型校验 + 环境变量注入，配置即数据模型；
- 关键参数（模型、路径、Top-K、开关）全部集中，便于做消融实验与复现。
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """全局配置。实例化一次，其他模块共享 settings 单例。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中与本系统无关的变量
    )

    # ---------- 模型服务 ----------
    dashscope_api_key: str = Field(
        validation_alias=AliasChoices("DASHSCOPE_API_KEY"),
        description="DashScope API Key",
    )
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("DASHSCOPE_BASE_URL"),
        description="OpenAI 兼容模式 Base URL",
    )
    chat_model: str = Field(
        default="qwen3.8-flash",
        validation_alias=AliasChoices("CHAT_MODEL", "MODEL_NAME"),
        description="对话 / Query 改写模型（兼容旧名 MODEL_NAME）",
    )
    embedding_model: str = Field(
        default="qwen3.7-text-embedding-flash",
        validation_alias=AliasChoices("EMBEDDING_MODEL"),
        description="文本向量化模型",
    )
    rerank_model: str = Field(
        default="qwen3.7-text-rerank",
        validation_alias=AliasChoices("RERANK_MODEL"),
        description="Rerank 重排序模型",
    )
    rerank_api_url: str = Field(
        default=(
            "https://dashscope.aliyuncs.com/api/v1/services/"
            "rerank/text-rerank/text-rerank"
        ),
        validation_alias=AliasChoices("RERANK_API_URL"),
        description="DashScope 原生 Rerank 接口",
    )

    # ---------- 路径 ----------
    kb_path: Path = Field(
        default=PROJECT_ROOT / "knowledge_base",
        validation_alias=AliasChoices("KB_PATH"),
        description="知识库文档目录",
    )
    index_path: Path = Field(
        default=PROJECT_ROOT / "faiss_index",
        validation_alias=AliasChoices("INDEX_PATH"),
        description="向量索引与 chunk 落盘目录",
    )

    # ---------- 切分与检索参数 ----------
    chunk_size: int = Field(default=600, ge=100, description="单块最大字符数")
    chunk_overlap: int = Field(default=100, ge=0, description="相邻块重叠字符数")
    retrieve_top_k: int = Field(
        default=20, ge=1, description="每路召回条数（BM25 / 向量各取 k 条）"
    )
    final_top_k: int = Field(default=4, ge=1, description="Rerank 后送入 LLM 的条数")
    rrf_k: int = Field(default=60, ge=1, description="RRF 融合常数，抑制头部排名过度加权")
    max_turns: int = Field(default=10, ge=1, description="保留的对话轮数")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="生成温度")

    # ---------- 功能开关（便于消融实验对比效果）----------
    enable_hybrid: bool = Field(default=True, description="是否启用 BM25 + 向量混合检索")
    enable_rerank: bool = Field(default=True, description="是否启用 Rerank 重排序")
    enable_query_rewrite: bool = Field(default=True, description="是否启用多轮 Query 改写")


# 模块级单例：其余模块 from config import settings
settings = Settings()

# 系统提示词单独定义为常量，不塞进 .env（长文本，避免转义问题）
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请严格遵守以下规则：
1. 仅依据提供的「参考资料」作答。资料中没有的信息，直接回答"根据现有资料无法回答"，绝对不要编造。
2. 引用到某段资料时，在该句末尾用方括号标注其编号，如 [1]、[2]；同一段可多处引用。
3. 回答简洁明了，直接回应问题，不要复述参考资料原文。"""

# Query 改写提示词：把带指代 / 省略的追问补成可独立检索的问题
QUERY_REWRITE_PROMPT = """你是检索查询改写助手。根据对话历史，把用户最新的追问改写成一个不依赖上下文、可独立在知识库中检索的问题。
要求：
- 把"它 / 这个 / 那 / 它们 / 上述 / 前者"等代词替换为具体对象；
- 补全被省略的主语或限定条件；
- 严格保持原意，不要引入原文没有的信息，也不要试图回答问题；
- 只输出改写后的问题本身，不要加"改写后："之类的前缀或任何解释。"""
