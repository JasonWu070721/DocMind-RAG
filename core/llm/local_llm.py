import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        logger.info("Initializing LLM")
        _llm =  ChatOpenAI(
            model="gpt-oss-120b",
            openai_api_key="local",
            base_url="http://192.168.60.190:8033/v1",
            temperature=0.3,
            request_timeout=120,
            max_tokens=1024
        )
        logger.info("LLM ready")
    return _llm