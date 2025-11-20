from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from engine import generate_answer
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="CookBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    timestamp: str


@app.get("/")
def health():
    return {"status": "CookBot API works"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = generate_answer(request.question)

    return ChatResponse(
        question=request.question,
        answer=answer,
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

