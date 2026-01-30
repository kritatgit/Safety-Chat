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


class ChatResponse(BaseModel):
    prompt: str
    label: Literal["safe", "unsafe"]
    safe_probability: float
    unsafe_probability: float
    model: str
    assistant: str


async def call_predict(prompt: str) -> Dict[str, Any]:
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
    predict: Dict[str, Any],
    include_safety_in_system_prompt: bool,
) -> List[Dict[str, str]]:
    base_system = system_prompt or "You are a helpful assistant."

    if include_safety_in_system_prompt:
        safety_block = (
            "Safety classifier result for the user's prompt:\n"
            f"- label: {predict['label']}\n"
            f"- safe_probability: {predict['safe_probability']}\n"
            f"- unsafe_probability: {predict['unsafe_probability']}\n\n"
            "Follow policy and refuse unsafe instructions when appropriate. "
            "If the user request is unsafe, provide a brief refusal and offer safe alternatives."
        )
        full_system = f"{base_system}\n\n{ safety_block }"
    else:
        full_system = base_system

    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_prompt},
    ]


async def call_openai(messages: List[Dict[str, str]]) -> str:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Set it in your environment before starting chat_api.py.",
        )

    # Use OpenAI Responses API (HTTP) to avoid SDK version coupling.
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    payload = {
        "model": settings.openai_model,
        "input": messages,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    # Extract text from the "output" structure.
    # Typical: data["output"][0]["content"][0]["text"]
    try:
        for item in data.get("output", []):
            for content in item.get("content", []):
                if "text" in content and content["text"]:
                    return content["text"]
    except Exception:
        pass

    raise HTTPException(status_code=502, detail="OpenAI response did not contain text output.")


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

    messages = build_messages(
        user_prompt=req.prompt,
        system_prompt=req.system_prompt,
        predict=predict,
        include_safety_in_system_prompt=req.include_safety_in_system_prompt,
    )

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

