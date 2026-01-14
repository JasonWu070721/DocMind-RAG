import logging
import os
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
_embedding = None


def get_embedding():
    """
    Use llama-server (GGUF) embedding endpoint instead of in-process HF model.
    """
    global _embedding

    if _embedding is None:
        logger.info("Initializing embedding via llama-server")

        base_url = os.environ.get(
            "EMBEDDING_BASE_URL",
            "http://192.168.80.190:8003/v1",
        )
        model_name = os.environ.get(
            "EMBEDDING_MODEL_NAME",
            "Qwen3-Embedding-8B",
        )

        _embedding = OpenAIEmbeddings(
            model=model_name,
            openai_api_key="local",
            base_url=base_url,
        )

        logger.info("Embedding model ready (llama-server)")

    return _embedding
