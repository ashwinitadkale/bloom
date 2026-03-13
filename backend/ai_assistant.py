import requests
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_ai(message, cycle_data):

    prompt = f"""
You are Bloom, an AI menstrual health assistant.

User cycle data:
Age: {cycle_data['age']}
Cycle Length: {cycle_data['cycle_length']}
Days Since Last Period: {cycle_data['last_period_day']}
Mood: {cycle_data['mood']}
Symptom: {cycle_data['symptom']}

User Question:
{message}

Give helpful menstrual health advice.
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages":[
                {"role":"user","content":prompt}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]