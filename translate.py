"""
============================================================
🤖 智能翻译 Agent (Smart Translator)
============================================================
这是一个基于 LangChain 和 阿里云百炼(Qwen) 大模型的翻译工具。
你可以直接在终端中连续输入句子进行翻译，也可以指定翻译的目标语言。

【运行前准备】
1. 获取 API Key：
   前往阿里云百炼大模型服务平台 (https://bailian.console.aliyun.com/) 注册并获取 API Key。
2. 配置环境变量：
   在 PyCharm 运行配置里，或者在系统环境变量中配置：
   QWEN_API_KEY=sk-你的真实api-key填在这里
3. 安装所需依赖库（在终端运行）：
   pip install langchain langchain-openai python-dotenv

【使用说明】
- 默认模式：直接输入句子，将自动翻译为【中文】。
- 指定模式：输入 `to:语言 句子`，例如输入 `to:English 今天天气很好`。
- 退出程序：输入 `quit` 或 `exit`。
============================================================
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


class SmartTranslator:
    def __init__(self):
        self.model = init_chat_model(
            "qwen3.8-flash",
            model_provider="openai",
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("QWEN_API_KEY"),
            temperature=0.3,
        )

    def translate(self, text: str, target_lang: str = "中文", source_lang: str = "自动检测") -> str:
        """
        将 text 翻译为目标语言。
        :param text: 待翻译文本
        :param target_lang: 目标语言，如 "中文"、"English"、"日本語"
        :param source_lang: 源语言，默认自动检测
        :return: 翻译结果
        """
        system_prompt = (
            "你是一个专业的多语言翻译引擎。要求：\n"
            "1. 准确传达原文含义，保持语气与风格；\n"
            "2. 只输出译文，不要解释、不要加引号、不要复述原文；\n"
            "3. 保留原文的格式（换行、列表、标点结构）；\n"
            "4. 专有名词、代码、URL 保持原样。"
        )
        user_prompt = (
            f"源语言：{source_lang}\n"
            f"目标语言：{target_lang}\n"
            f"待翻译内容：\n{text}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.model.invoke(messages)
        return response.content.strip()

    def translate_batch(self, texts: list[str], target_lang: str = "中文", source_lang: str = "自动检测") -> list[str]:
        """批量翻译，逐条调用（简单可靠）。"""
        return [self.translate(t, target_lang, source_lang) for t in texts]


if __name__ == "__main__":
    translator = SmartTranslator()
    print("=" * 60)
    print("🤖 欢迎使用智能翻译 Agent！")
    print("💡 【默认模式】直接输入内容，将自动翻译为【中文】。")
    print("💡 【指定语言】如果要翻译成其他语言，请在句子前加上 'to:语言'。")
    print("    例如：to:English 今天天气很好   -> 输出英文")
    print("    例如：to:Japanese 你好         -> 输出日文")
    print("💡 输入 'quit' 或 'exit' 退出程序。")
    print("=" * 60 + "\n")

    while True:
        user_input = input("\n请输入要翻译的句子 (或输入 'quit' 退出): ").strip()

        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            continue

        # 简单的指令解析：如果输入以 "to:English " 开头
        if user_input.startswith("to:"):
            try:
                # 提取目标语言和真正的内容
                parts = user_input.split(" ", 1)
                target = parts[0].replace("to:", "")
                content = parts[1]

                print(f"正在翻译为【{target}】...")
                result = translator.translate(content, target_lang=target)
                print(f"【翻译结果】:\n{result}")
            except IndexError:
                print("格式错误，请用 'to:目标语言 句子' 的格式，例如：to:English 你好")
        else:
            # 默认翻译成中文
            result = translator.translate(user_input, target_lang="中文")
            print(f"【翻译结果】:\n{result}")