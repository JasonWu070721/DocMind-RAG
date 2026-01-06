from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="gpt-oss-120b",
        openai_api_key="local",
        base_url="http://192.168.60.190:8033/v1",
        temperature=0.3,
        request_timeout=120,
        max_tokens=1024
    )