from langchain.chat_models import init_chat_model
from rich import print as rprint
##统一使用init_chat_model（）完成模型的初始化
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


#1 读取，env文件中的配置信息，相关的环境变量以.env文件中的优先
load_dotenv(override=True)
my_api_key=os.getenv("API_KEY")
my_base_url=os.getenv("BASE_URL")
#2 初始化大模型
model= init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=my_api_key,
    base_url=my_base_url,
    timeout=30,
    max_retries=3,
)
#3 声明1个函数（工具）
def get_weather(city:str):
    return f"{city}当天天气清理"
#4 将函数绑定在模型上
model_with_tools=model.bind_tools(
    [get_weather]
)
#5 调用模型
response=model_with_tools.invoke("北京的天气怎么样")
rprint(response)