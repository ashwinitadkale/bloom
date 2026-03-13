# backend/models.py
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Example model
from sqlalchemy import Column, Integer, String

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)