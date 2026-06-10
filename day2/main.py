from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set.")

client = Groq(api_key=api_key)
app = FastAPI()

# stores conversation history for each session
# key = session_id, value = list of messages
sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # optional — if not provided we create a new one

@app.get("/")
def root():
    return {"status": "AI chat API is running"}

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    sessions[session_id].append({
        "role": "user",
        "content": request.message
    })

    def generate():
        reply = ""
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=sessions[session_id],
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                reply += token
                yield token

        # after stream ends, save full reply to history
        sessions[session_id].append({
            "role": "assistant",
            "content": reply
        })

    return StreamingResponse(generate(), media_type="text/plain")