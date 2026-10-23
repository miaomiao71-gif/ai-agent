"""
多功能 Agent，包含三个工具：
1. 计算
2. 获取天气
3. 获取温度
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()


# ============ 定义工具 ============

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如 '2 + 3 * 4' 或 '(10 - 2) / 4'。
    支持 + - * / ** 和括号。"""
    try:
        # 安全起见，只允许数字和运算符
        allowed = set("0123456789+-*/().% ")
        if not set(expression) <= allowed:
            return "表达式包含非法字符，只支持数字和 + - * / % ( )"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气情况（模拟数据）。
    参数 city: 城市名，例如 '北京'。"""
    # 这里用模拟数据，实际可替换为真实天气 API
    mock_weather = {
        "北京": "晴，微风，能见度良好",
        "上海": "多云转阴，有小雨",
        "广州": "雷阵雨",
        "深圳": "晴间多云",
        "杭州": "阴天",
    }
    return mock_weather.get(city, f"{city}：晴，空气良好（模拟数据）")


@tool
def get_temperature(city: str) -> str:
    """获取指定城市的当前温度（摄氏度，模拟数据）。
    参数 city: 城市名，例如 '北京'。"""
    mock_temp = {
        "北京": 18,
        "上海": 22,
        "广州": 28,
        "深圳": 27,
        "杭州": 20,
    }
    temp = mock_temp.get(city, 25)
    return f"{city} 当前温度：{temp}°C（模拟数据）"


# ============ 初始化 Agent ============

def init():
    """初始化模型和 Agent"""
    model = init_chat_model(
         model="qwen3.8-flash",
        model_provider="openai",
         base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
         api_key=os.getenv("QWEN_API_KEY"),
         temperature=0.1,
    )

    tools = [calculate, get_weather, get_temperature]

    system_prompt = (
        "你是一个多功能助手，可以使用以下工具：\n"
        "- calculate: 执行数学计算\n"
        "- get_weather: 查询城市天气\n"
        "- get_temperature: 查询城市温度\n"
        "根据用户问题选择合适的工具，必要时可以组合使用多个工具。"
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


# ============ 运行示例 ============

if __name__ == "__main__":
    agent = init()

    queries = [
        "帮我算一下 (12 + 8) * 3 是多少",
        "北京现在天气怎么样？温度多少？",
        "上海和广州哪个更热？",
    ]

    for q in queries:
        print(f"\n{'='*60}\n用户: {q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        print(f"助手: {result['messages'][-1].content}")
    history = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        history.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": history})
            # 用返回的完整消息列表更新历史（包含工具调用、工具结果、最终回答）
            history = result["messages"]
            answer = history[-1].content
            # 兼容 Qwen 可能返回的列表形式内容
            if isinstance(answer, list):
                answer = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in answer
                )
            print(f"助手: {answer}")
        except Exception as e:
            print(f"出错了: {e}")
            # 出错时回退这次用户输入，避免污染历史
            history.pop()