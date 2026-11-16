import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
MODEL_NAME= os.getenv("MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
INDEX_PATH = os.getenv("INDEX_PATH")

if not API_KEY or not BASE_URL:
    sys.exit("错误：请在 .env 中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")

# ================= 配置区 =================
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。
请严格根据提供的参考文档回答用户的问题。
如果参考文档中找不到答案，请直接回答“根据现有资料无法回答”，绝对不要编造。
回答时，尽量简洁明了。"""
MAX_TURNS = 10
TEMPERATURE = 0.1
# ==========================================


class ChatAgent:
    def __init__(self):
        self.model = self.init_model()
        self.retriever = self.init_retriever()
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]

    def init_model(self):
        return init_chat_model(
            MODEL_NAME,
            model_provider="openai",
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE,
        )

    def init_retriever(self):
        if not os.path.exists(INDEX_PATH):
            sys.exit(f"错误：找不到向量库 {INDEX_PATH}。请先运行 python build_db.py 构建知识库。")

        print("正在加载本地向量数据库...")
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            check_embedding_ctx_length=False
        )

        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore.as_retriever(search_kwargs={"k": 4})

    def _trim_history(self):
        if MAX_TURNS <= 0:
            return
        system_msg = self.history[0]
        dialog = self.history[1:]
        max_msgs = MAX_TURNS * 2
        if len(dialog) > max_msgs:
            self.history = [system_msg] + dialog[-max_msgs:]

    def send_stream(self, user_input: str):
        docs = self.retriever.invoke(user_input)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([os.path.basename(d.metadata.get("source", "未知来源")) for d in docs]))
            source_text = "、".join(sources)
            augmented_input = f"参考资料：\n{context}\n\n用户问题：{user_input}"
        else:
            augmented_input = user_input
            source_text = "无"

        self.history.append(HumanMessage(content=augmented_input))
        self._trim_history()

        full_response = ""
        try:
            for chunk in self.model.stream(self.history):
                content = chunk.content
                if content:
                    full_response += content
                    yield content
        finally:
            if full_response:
                self.history.append(AIMessage(content=full_response))

def main():
    agent = ChatAgent()
    print("知识库对话助手已启动（流式输出），输入 quit 退出。\n")
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
            # 接收生成器的内容并实时打印
            for chunk in agent.send_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")  # 对话结束后换行
        except Exception as e:
            print(f"\n[请求出错] {e}\n")


if __name__ == "__main__":
    main()