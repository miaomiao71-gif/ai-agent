"""
============================================================
🤖 智能文案生成 Agent (Smart Copywriter)
============================================================
这是一个基于 LangChain 和 阿里云百炼(Qwen) 大模型的结构化文案生成工具。
支持生成多种场景、多种语气的文案，并自带五维评分系统。

【运行前准备】
1. 获取 API Key：
   前往阿里云百炼大模型服务平台 (https://bailian.console.aliyun.com/) 注册并获取 API Key。
2. 配置环境变量：
   在 PyCharm 运行配置里，或者在系统环境变量中配置：
   QWEN_API_KEY=sk-你的真实api-key填在这里
3. 安装所需依赖库（在终端运行）：
   pip install langchain langchain-openai pydantic
============================================================
"""

import os
from typing import List, Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# 1. 定义结构化输出模型 (Pydantic)
class CopywritingScore(BaseModel):
    """文案评分维度"""
    creativity: int = Field(description="创意性，1-10分")
    clarity: int = Field(description="清晰度，1-10分")
    appeal: int = Field(description="吸引力，1-10分")
    emotion: int = Field(description="情感共鸣，1-10分")
    conciseness: int = Field(description="简洁性，1-10分")


class Copywriting(BaseModel):
    """单条文案结果"""
    content: str = Field(description="文案内容")
    score: CopywritingScore = Field(description="质量评分")


class CopywritingBatch(BaseModel):
    """批量文案结果"""
    copies: List[Copywriting] = Field(description="文案列表")


# ==========================================
# 2. 定义智能文案生成器 Agent
# ==========================================
class SmartCopywriter:
    """智能文案生成器"""

    def __init__(self):
        self.model = init_chat_model(
            model="qwen3.8-flash",
            model_provider="openai",
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("QWEN_API_KEY"),  # 这里现在能成功读到了
            temperature=0.9,
        )
        # 绑定结构化输出
        self.structured_model = self.model.with_structured_output(CopywritingBatch)

    def generate(
            self,
            product: str,
            copy_type: Literal["广告", "社交媒体", "邮件", "博客", "短视频", "产品描述"],
            tone: Literal["正式", "轻松", "幽默", "专业"],
            count: int = 3
    ) -> List[Copywriting]:
        """
        生成文案
        """
        prompt = f"""
        你是一位顶级的专业文案专家。请根据以下要求为产品创作文案：

        产品/服务信息：{product}
        文案类型：{copy_type}
        语气风格：{tone}
        生成数量：{count} 条

        要求：
        1. 必须生成 {count} 条不同角度的文案。
        2. 请为每一条文案的五个维度（创意性、清晰度、吸引力、情感共鸣、简洁性）进行打分（1-10分）。
        3. 严格按照要求的格式输出。
        """

        print(f"⏳ 正在生成【{tone}】风格的【{copy_type}】文案...\n")

        # 调用模型，返回的已经是 CopywritingBatch 对象
        result = self.structured_model.invoke(prompt)

        return result.copies


# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    # 实例化 Agent
    agent = SmartCopywriter()

    # 测试生成
    try:
        results = agent.generate(
            product="一款主打极简设计、降噪功能的真无线蓝牙耳机，续航30小时",
            copy_type="社交媒体",
            tone="轻松",
            count=2
        )

        # 打印结果
        for i, copy_item in enumerate(results, 1):
            print(f"--- 文案 {i} ---")
            print(f"内容：{copy_item.content}")
            print(f"评分：{copy_item.score.model_dump()}")
            print()

    except Exception as e:
        print(f"运行出错: {e}")