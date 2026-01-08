import logging
from core.logging.logger import setup_logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from apps.api.routes import rag
from apps.api.routes import openai
from core.llm.local_llm import get_llm
from core.vectorstore.chroma_store import get_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== logging =====
    setup_logging(level=logging.INFO)

    logging.info("API startup: preload models")
    # ===== startup =====
    logging.info(">>> API startup: preload models")
    get_vectorstore()
    get_llm()
    logging.info(">>> API ready")

    yield 

    # ===== shutdown =====
    logging.info(">>> API shutdown: cleanup (if needed)")


app = FastAPI(
    title="DocMind RAG API",
    lifespan=lifespan,
)

app.include_router(rag.router)
app.include_router(openai.router)


@app.get("/health")
def health():
    return {"status": "ok"}
