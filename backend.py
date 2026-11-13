from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from main import ChatbotAssistant, get_stocks

# Ensure NLTK data is downloaded on the server
nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

# Initialize and load the saved model and dimensions
assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()
assistant.load_model('chatbot_model.pth', 'dimensions.json')

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    reply = assistant.process_message(request.message)
    return {"reply": reply}