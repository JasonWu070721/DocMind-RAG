import json
import logging
import time
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.embeddings.qwen_embedding import get_embedding
from core.rag.service import answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI"])


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    user: Optional[str] = None


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Any
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _extract_user_prompt(messages: List[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            content = _normalize_content(msg.content)
            if content:
                return content
    raise HTTPException(status_code=400, detail="No user message found in messages.")

def _chunk_text(text: str, size: int = 20) -> List[str]:
    if size <= 0:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]


@router.get("/models")
def list_models():
    from core.llm.local_llm import get_llm

    llm = get_llm()
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "local")
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": now,
                "owned_by": "local",
            }
        ],
    }


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(payload: ChatCompletionRequest):
    prompt = _extract_user_prompt(payload.messages)
    logger.debug("OpenAI chat completions prompt: %s", prompt)

    response_text = answer(prompt)
    model_name = payload.model or "local"
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    choice_count = max(int(payload.n or 1), 1)

    if payload.stream:
        def event_stream():
            for chunk in _chunk_text(response_text):
                chunk_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_payload, ensure_ascii=True)}\n\n"

            final_payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    choices = [
        ChatCompletionChoice(
            index=i,
            message=ChatCompletionMessage(role="assistant", content=response_text),
            finish_reason="stop",
        )
        for i in range(choice_count)
    ]

    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=model_name,
        choices=choices,
        usage=usage,
    )


@router.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(payload: EmbeddingRequest):
    embedding_model = get_embedding()
    inputs = payload.input

    if isinstance(inputs, str):
        texts = [inputs]
    elif isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
        texts = inputs
    else:
        raise HTTPException(status_code=400, detail="Input must be a string or list of strings.")

    vectors = embedding_model.embed_documents(texts)

    data = [
        EmbeddingData(embedding=vector, index=i)
        for i, vector in enumerate(vectors)
    ]
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return EmbeddingResponse(
        data=data,
        model=payload.model or "local",
        usage=usage,
    )
