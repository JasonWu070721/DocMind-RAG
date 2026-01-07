from core.loaders.ocr_json_loader import load_ocr_json
from core.chunkers.recursive_chunker import chunk_documents
from core.vectorstore.chroma_store import get_vectorstore

def ingest_ocr_json(path: str):
    docs = load_ocr_json(path)
    chunks = chunk_documents(docs)

    vs = get_vectorstore()
    vs.add_documents(chunks)

if __name__ == "__main__":
    ingest_ocr_json(
        "D:/github/DocMind-RAG/data/PrintingTechniqueDataset_json/print-tech-outlier-01-001_res.json"
    )
