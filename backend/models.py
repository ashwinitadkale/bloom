from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date

# Base class for all models
Base = declarative_base()

# Example User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    birthdate = Column(Date, nullable=True)