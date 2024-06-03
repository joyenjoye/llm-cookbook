import os
from dotenv import load_dotenv
from dotenv import find_dotenv


# 总的来说，这段代码通过组合使用 dotenv 库和 os 模块实现了一个简洁而高效地从 .env 文件加载并获取 OpenAI API KEY 的解决方案。
# 这对于保护敏感信息和提高代码可移植性都非常有益。


def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    # 调用之前定义的 load_env() 函数来确保环境变量已经从 .env 文件加载进来。
    load_env()
    # 然后使用之前导入的 os 模块中的 getenv 方法获取名为 "OPENAI_API_KEY" 的环境变量值。
    openai_api_key = os.getenv("OPENAI_API_KEY")

    return openai_api_key
