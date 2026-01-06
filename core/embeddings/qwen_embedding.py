from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-8B",
        encode_kwargs={"normalize_embeddings": True}
    )