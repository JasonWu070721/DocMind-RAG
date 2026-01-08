import logging
import os
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        logger.info("Initializing LLM")
        model = os.environ.get("LLM_MODEL", "gpt-oss-120b")
        base_url = os.environ.get("OPENAI_BASE_URL", "http://192.168.60.190:8033/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "local")
        temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
        request_timeout = int(os.environ.get("LLM_REQUEST_TIMEOUT", "120"))
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
        _llm =  ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=max_tokens
        )
        logger.info("LLM ready")
    return _llm
