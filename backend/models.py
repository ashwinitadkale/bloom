from sqlalchemy import Column, Integer, Float
from .database import Base

class CycleLog(Base):
    __tablename__ = "cycle_logs"

    id = Column(Integer, primary_key=True, index=True)

    age = Column(Integer)
    cycle_length = Column(Integer)
    days_since_last = Column(Integer)

    mood = Column(Integer)
    flow_intensity = Column(Integer)
    symptom = Column(Integer)

    stress = Column(Integer)
    sleep = Column(Integer)
    exercise = Column(Integer)

    bmi = Column(Float)
    avg_cycle = Column(Float)
    cycle_var = Column(Float)
