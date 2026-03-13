from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# PostgreSQL connection URL
# Replace these values with your own credentials
DATABASE_URL = "postgresql+psycopg2://postgres:candy02@localhost:5432/bloom"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)