from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modified import ACEAdvisoryLegalChatbot
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ACE Advisory Legal Chatbot API")

# Initialize the chatbot
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable is required")

chatbot = ACEAdvisoryLegalChatbot(mistral_api_key=mistral_api_key)

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query):
    try:
        response = chatbot.generate_response(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 