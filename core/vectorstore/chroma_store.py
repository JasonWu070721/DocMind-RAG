import os
import logging
from chromadb import HttpClient
from langchain_chroma import Chroma
from core.embeddings.qwen_embedding import get_embedding

logger = logging.getLogger(__name__)

def get_vectorstore() -> Chroma:
    """
    Get Chroma vectorstore via HTTP client (remote Chroma server)
    """

    logger.info("Connecting to remote Chroma server")

    client = HttpClient(
        host=os.getenv("CHROMA_HOST", "192.168.60.190"),
        port=int(os.getenv("CHROMA_PORT", "8010")),
    )

    vectorstore = Chroma(
        client=client,
        collection_name=os.getenv("CHROMA_COLLECTION", "docmind_db"),
        embedding_function=get_embedding(),
    )

    logger.info("Chroma vectorstore ready (remote)")
    return vectorstore
