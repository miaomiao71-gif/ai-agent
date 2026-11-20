"""在线问答链路：Query 改写 → 混合检索 → 组装带引用的上下文 → 流式生成。

修复的关键 bug（相比旧版）：
- 旧版把「参考资料 + 用户问题」拼成的 augmented_input 存入对话历史，
  导致每轮历史都携带上一轮的检索结果，历史雪球式膨胀且模型会误把旧资料当上下文。
- 现在：history 只存原始用户输入与模型回答；检索到的资料只在当前轮临时注入模型，
  不进入历史。Query 改写也只作用于检索词，不污染原始提问。

溯源设计：
- 每个检索 chunk 分配引用编号 [1][2]…，注入上下文时带上文件名 / 页码 / 块号；
- 系统提示词要求模型在引用处标注编号，末尾再输出「参考来源」清单（段落级定位）。
"""

import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT, settings
from retrieval import HybridRetriever


class ChatAgent:
    def __init__(self) -> None:
        self.model = self._init_model()
        try:
            self.retriever = HybridRetriever()
        except FileNotFoundError as e:
            sys.exit(f"错误：{e}")
        # 对话历史只存「原始问答」，绝不放检索到的参考资料
        self.history: list = [SystemMessage(content=SYSTEM_PROMPT)]

    def _init_model(self):
        return init_chat_model(
            settings.chat_model,
            model_provider="openai",
            base_url=settings.dashscope_base_url,
            api_key=settings.dashscope_api_key,
            temperature=settings.temperature,
        )

    def _trim_history(self) -> None:
        """按轮数裁剪历史，保留 system 消息。"""
        system_msg, dialog = self.history[0], self.history[1:]
        max_msgs = settings.max_turns * 2
        if len(dialog) > max_msgs:
            self.history = [system_msg] + dialog[-max_msgs:]

    def _recent_dialog_text(self, max_msgs: int = 4) -> str:
        """取最近若干条对话，拼成给 Query 改写用的文本。"""
        lines = []
        for msg in self.history[1:][-max_msgs:]:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}：{msg.content}")
        return "\n".join(lines)

    def rewrite_query(self, user_input: str) -> str:
        """多轮指代消解：把「它 / 这个 / 上面那个」等补成可独立检索的问题。

        仅在存在多轮对话且开关打开时调用；失败则回退到原始输入，保证可用性。
        """
        if not settings.enable_query_rewrite or len(self.history) <= 1:
            return user_input
        try:
            messages = [
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(
                    content=(
                        f"对话历史：\n{self._recent_dialog_text()}\n\n"
                        f"用户最新追问：{user_input}\n\n改写后的问题："
                    )
                ),
            ]
            resp = self.model.invoke(messages)
            rewritten = resp.content.strip().strip('"').strip("'")
            return rewritten or user_input
        except Exception as e:
            print(f"[提示] Query 改写失败，使用原始问题：{e}")
            return user_input

    @staticmethod
    def _format_context(docs: list) -> tuple[str, list[str]]:
        """给每个 chunk 编号并附段落级来源，返回（上下文, 来源清单）。"""
        context_parts: list[str] = []
        source_lines: list[str] = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            page = meta.get("page", -1)
            location = f"第{page}页" if isinstance(page, int) and page >= 0 else "无页码"
            source_lines.append(
                f"  [{i}] {meta.get('file_name', '未知文件')} · {location} · "
                f"第{meta.get('chunk_index', 0) + 1}段 · "
                f"相关度 {meta.get('rerank_score', '-')}"
            )
            header = (
                f"[{i}] 来源：{meta.get('file_name', '未知文件')}（{location}，"
                f"第{meta.get('chunk_index', 0) + 1}段）"
            )
            context_parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(context_parts), source_lines

    def send_stream(self, user_input: str):
        """流式回答。yield 给用户的内容片段；检索来源通过 self.last_sources 暴露。"""
        # 1. Query 改写（只用于检索）
        search_query = self.rewrite_query(user_input)
        self.last_sources: list[str] = []

        # 2. 混合检索
        docs = self.retriever.retrieve(search_query)
        if docs:
            context, self.last_sources = self._format_context(docs)
            augmented_input = (
                f"参考资料：\n{context}\n\n用户问题：{user_input}"
            )
        else:
            augmented_input = user_input

        # 3. 组装本轮消息：system + 干净历史 + 本轮增强输入
        #    注意 augmented_input 不写入 self.history，杜绝历史污染
        turn_messages = (
            [self.history[0], *self.history[1:], HumanMessage(content=augmented_input)]
        )

        full_response = ""
        try:
            for chunk in self.model.stream(turn_messages):
                content = chunk.content
                if content:
                    full_response += content
                    yield content
        finally:
            # 4. 只把原始输入与回答写入历史
            if full_response:
                self.history.append(HumanMessage(content=user_input))
                self.history.append(AIMessage(content=full_response))
                self._trim_history()


def main() -> None:
    agent = ChatAgent()
    print("知识库对话助手已启动（混合检索 + Rerank + 流式输出），输入 quit 退出。\n")
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        if not user_input:
            continue

        try:
            print("助手：", end="", flush=True)
            for chunk in agent.send_stream(user_input):
                print(chunk, end="", flush=True)
            if agent.last_sources:
                print("\n\n参考来源：")
                print("\n".join(agent.last_sources))
            print("\n")
        except Exception as e:
            print(f"\n[请求出错] {e}\n")


if __name__ == "__main__":
    main()
