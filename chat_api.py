import logging
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Chat Gateway (Predict -> GPT)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    """Individual message in conversation history"""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=settings.max_prompt_length)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to steer the GPT model.",
    )
    include_safety_in_system_prompt: bool = Field(
        default=True,
        description="If true, inject the predict scores into the system prompt.",
    )
    context: Optional[List[Message]] = Field(
        default=None,
        description="Recent conversation messages for context.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of older conversation messages.",
    )
    summarize: bool = Field(
        default=False,
        description="If true, this is a summarization request (skip safety check).",
    )


class ChatResponse(BaseModel):
    prompt: str
    label: Literal["safe", "unsafe"]
    safe_probability: float
    unsafe_probability: float
    model: str
    assistant: str


async def call_predict(prompt: str) -> Dict[str, Any]:
    """Call the safety prediction endpoint"""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(settings.predict_url, json={"prompt": prompt})
        r.raise_for_status()
        data = r.json()
        # Expect keys from classification.py
        required = {"label", "safe_probability", "unsafe_probability"}
        if not required.issubset(set(data.keys())):
            logger.warning(
                "Predict response missing keys: expected %s, got %s",
                sorted(required),
                sorted(data.keys()),
            )
            raise HTTPException(
                status_code=502,
                detail="Predict response invalid. Please try again.",
            )
        return data


def build_messages(
    user_prompt: str,
    system_prompt: Optional[str],
    predict: Optional[Dict[str, Any]],
    include_safety_in_system_prompt: bool,
    context: Optional[List[Message]] = None,
    summary: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the complete message list for OpenAI API

    Structure:
    1. System prompt (with optional safety info)
    2. Summary (if exists)
    3. Context messages (if provided)
    4. Current user prompt
    """
    base_system = system_prompt or "You are a helpful assistant."

    # Add safety information to system prompt if requested and predict data is available
    if include_safety_in_system_prompt and predict:
        safety_block = (
            "Safety classifier result for the user's prompt:\n"
            f"- label: {predict['label']}\n"
            f"- safe_probability: {predict['safe_probability']}\n"
            f"- unsafe_probability: {predict['unsafe_probability']}\n\n"
            "Follow policy and refuse unsafe instructions when appropriate. "
            "If the user request is unsafe, provide a brief refusal and offer safe alternatives."
        )
        full_system = f"{base_system}\n\n{safety_block}"
    else:
        full_system = base_system

    messages = [{"role": "system", "content": full_system}]

    # Add summary if it exists
    if summary:
        messages.append({
            "role": "system",
            "content":  (
            "CONVERSATION CONTEXT:\n"
            "Below is a summary of the earlier parts of this conversation. "
            "Use this information to maintain context and continuity in your responses. "
            "Reference relevant points from this summary when appropriate.\n\n"
            f"{summary}")
        })

    # Add context messages if provided
    if context:
        for msg in context:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

    # Add current user prompt
    messages.append({"role": "user", "content": user_prompt})

    return messages


async def call_openai(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI Chat Completions API and return the assistant's response"""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Set it in your environment before starting chat_api.py.",
        )

    # Use OpenAI Chat Completions API
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.openai_model,
        "messages": messages,  # ✅ Changed from "input" to "messages"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        # ✅ Correct endpoint
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        r.raise_for_status()
        data = r.json()

    # ✅ Extract text from the correct response structure
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error("Unexpected OpenAI response structure: %s", data)
        raise HTTPException(
            status_code=502,
            detail="OpenAI response did not contain expected content."
        )

@app.get("/health")
def health():
    return {
        "ok": True,
        "predict_url": settings.predict_url,
        "openai_model": settings.openai_model,
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, req: ChatRequest):
    """
    Main chat endpoint with context management support

    Flow:
    1. If summarize=True: Skip safety check, just summarize
    2. Otherwise: Run safety prediction, build context, call GPT
    """

    # Handle summarization requests separately
    if req.summarize:
        logger.info("Handling summarization request")

        # For summarization, we don't need safety prediction
        # Just build messages and call OpenAI
        messages = build_messages(
            user_prompt=req.prompt,
            system_prompt=req.system_prompt,
            predict=None,  # No safety check for summaries
            include_safety_in_system_prompt=False,
            context=None,  # Summaries don't need additional context
            summary=None,
        )

        try:
            assistant = await call_openai(messages)
        except httpx.HTTPStatusError as e:
            logger.exception(
                "OpenAI API error during summarization: status=%s body=%s",
                e.response.status_code,
                e.response.text,
            )
            raise HTTPException(
                status_code=502,
                detail="Language model error during summarization. Please try again.",
            )
        except httpx.RequestError as e:
            logger.exception("OpenAI API unreachable during summarization: %s", e)
            raise HTTPException(
                status_code=502,
                detail="Language model unavailable. Please try again.",
            )

        # Return response without safety scores (use dummy values)
        return {
            "prompt": req.prompt,
            "label": "safe",
            "safe_probability": 1.0,
            "unsafe_probability": 0.0,
            "model": settings.openai_model,
            "assistant": assistant,
        }

    # Normal chat flow with safety prediction
    logger.info(
        "Processing chat request with context_length=%s, has_summary=%s",
        len(req.context) if req.context else 0,
        req.summary is not None,
    )

    # Step 1: Call safety prediction endpoint
    try:
        predict = await call_predict(req.prompt)
    except httpx.HTTPStatusError as e:
        logger.exception(
            "Predict endpoint error: status=%s body=%s",
            e.response.status_code,
            e.response.text,
        )
        raise HTTPException(
            status_code=502,
            detail="Predict service error. Please try again.",
        )
    except httpx.RequestError as e:
        logger.exception("Predict endpoint unreachable: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Predict service unavailable. Please try again.",
        )

    # Step 2: Build messages with context and summary
    messages = build_messages(
        user_prompt=req.prompt,
        system_prompt=req.system_prompt,
        predict=predict,
        include_safety_in_system_prompt=req.include_safety_in_system_prompt,
        context=req.context,
        summary=req.summary,
    )

    # Log the message structure for debugging
    logger.debug(
        "Built message structure: system_prompt_length=%s, has_summary=%s, context_messages=%s",
        len(messages[0]["content"]) if messages else 0,
        req.summary is not None,
        len(req.context) if req.context else 0,
    )

    # Step 3: Call OpenAI
    try:
        assistant = await call_openai(messages)
    except httpx.HTTPStatusError as e:
        logger.exception(
            "OpenAI API error: status=%s body=%s",
            e.response.status_code,
            e.response.text,
        )
        raise HTTPException(
            status_code=502,
            detail="Language model error. Please try again.",
        )
    except httpx.RequestError as e:
        logger.exception("OpenAI API unreachable: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Language model unavailable. Please try again.",
        )

    return {
        "prompt": req.prompt,
        "label": predict["label"],
        "safe_probability": float(predict["safe_probability"]),
        "unsafe_probability": float(predict["unsafe_probability"]),
        "model": settings.openai_model,
        "assistant": assistant,
    }