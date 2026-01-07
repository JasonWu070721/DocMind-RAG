import logging
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
_embedding = None

def get_embedding():
    global _embedding
    
    if _embedding is None:
        logger.info("Initializing embedding")
        _embedding = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-8B",
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model ready")
    return _embedding