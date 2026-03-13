from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import engine, SessionLocal
from .models import Base, User
from pydantic import BaseModel
from .ai_assistant import ask_ai

app = FastAPI(title="Bloom API with AI Assistant")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/ai/ask/")
def ai_chat(req: PromptRequest):
    """
    Accepts a prompt and returns the Groq AI assistant response.
    """
    answer = ask_ai(req.prompt)
    return {"response": answer}
conn = engine.connect()
print("Connected!")  # Should print without error
conn.close()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI with PostgreSQL"}

# Create a user
@app.post("/users/")
def create_user(name: str, email: str, db: Session = Depends(get_db)):
    user = User(name=name, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# List all users
@app.get("/users/")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()