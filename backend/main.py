# backend/main.py
import asyncio
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import engine, SessionLocal
from .models import Base, User
from pydantic import BaseModel
from .ai_assistant import ask_ai

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bloom API with AI Assistant")

# -------------------------
# AI Assistant
# -------------------------
class PromptRequest(BaseModel):
    prompt: str

@app.post("/ai/ask/")
async def ai_ask(request: PromptRequest):
    prompt = request.prompt
    loop = asyncio.get_event_loop()
    # Run blocking AI call in a thread
    result = await loop.run_in_executor(None, ask_ai, prompt)
    return {"response": result}

# -------------------------
# Database dependency
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# Test root
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Hello FastAPI with PostgreSQL"}

# -------------------------
# User CRUD
# -------------------------
@app.post("/users/")
def create_user(name: str, email: str, db: Session = Depends(get_db)):
    user = User(name=name, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.get("/users/")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()