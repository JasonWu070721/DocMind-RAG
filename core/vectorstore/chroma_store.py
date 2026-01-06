from langchain_chroma import Chroma
from core.embeddings.qwen_embedding import get_embedding

def get_vectorstore():
    return Chroma(
        collection_name="docmind_db",
        embedding_function=get_embedding(),
        persist_directory="./data/chroma_langchain_db"
    )
