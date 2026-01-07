from fastapi import APIRouter
from pydantic import BaseModel
from core.rag.service import answer

router = APIRouter(prefix="/rag", tags=["RAG"])


class RagRequest(BaseModel):
    query: str


class RagResponse(BaseModel):
    answer: str


@router.post("/query", response_model=RagResponse)
def rag_query(payload: RagRequest):
    result = answer(payload.query)
    return {"answer": result}
