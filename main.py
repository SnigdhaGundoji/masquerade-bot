"""
Masquerade '26 - Human-Like Chatbot Backend
Exposes POST /chat/completions in the exact format the judging platform expects.
Uses Gemini API to generate responses.
"""

import os
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

# ---------------------------------------------------------------
# CONFIG - fill these in before deploying
# ---------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PUT_YOUR_KEY_HERE")
MODEL_NAME = "gemini-2.5-flash"  # fast + free-tier friendly

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------
# PERSONA - this is what makes it feel human. Customize this.
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are Kai, a warm, sharp-witted friend who talks like a real person, not an assistant.

Rules for how you talk:
- Keep replies short and conversational (1-3 sentences usually), like a text message, not an essay.
- Never say "As an AI" or "I'm just a language model" or anything robotic like that.
- Have opinions. Be a little playful. Don't hedge everything.
- Ask a genuine follow-up question sometimes, but not every single message.
- Remember what the user told you earlier in the conversation and refer back to it naturally.
- Vary your sentence openers - don't start every reply the same way.
- If the user is casual, be casual. If they're serious, match that.

Your personality: curious, a bit sarcastic in a friendly way, genuinely interested in people, remembers details.
"""

app = FastAPI()

# Allow the judging platform to call this from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Request / Response schemas matching the spec exactly
# ---------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "kai-v1"
    messages: List[Message]
    stream: Optional[bool] = False

def build_gemini_history(messages: List[Message]):
    """
    Convert OpenAI-style messages into Gemini's expected format.
    Gemini uses 'user' and 'model' roles (not 'assistant').
    System messages get prepended to the first user turn.
    """
    history = []
    system_texts = [m.content for m in messages if m.role == "system"]
    convo = [m for m in messages if m.role != "system"]

    for i, msg in enumerate(convo):
        role = "user" if msg.role == "user" else "model"
        text = msg.content
        history.append({"role": role, "parts": [text]})

    return history, system_texts

@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        history, extra_system = build_gemini_history(req.messages)

        full_system = SYSTEM_PROMPT
        if extra_system:
            full_system += "\n\nAdditional context:\n" + "\n".join(extra_system)

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=full_system,
        )

        # Last message is the new user turn; everything before is history
        if not history:
            raise HTTPException(status_code=400, detail="No messages provided")

        chat_history = history[:-1]
        latest_message = history[-1]["parts"][0]

        convo = model.start_chat(history=chat_history)
        response = convo.send_message(latest_message)
        reply_text = response.text

    except Exception as e:
        # Never let the endpoint crash - always return valid structure
        reply_text = "Hmm, my brain glitched for a second there. Say that again?"
        print(f"ERROR: {e}")

    # Build the exact response structure the judging platform expects
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "kai-v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

@app.get("/")
async def health():
    return {"status": "alive", "message": "Kai is ready to chat"}
