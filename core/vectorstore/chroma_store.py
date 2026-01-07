import logging
from langchain_chroma import Chroma
from core.embeddings.qwen_embedding import get_embedding

logger = logging.getLogger(__name__)
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("Initializing Chroma vectorstore")
        _vectorstore=  Chroma(
            collection_name="docmind_db",
            embedding_function=get_embedding(),
            persist_directory="./data/chroma_langchain_db"
        )
        logger.info("Vectorstore ready")
    return _vectorstore
