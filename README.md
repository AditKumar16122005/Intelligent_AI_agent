# 🤖 PyTorch Intent-Based Chatbot

A lightweight, intent-classification chatbot built with **Python**, **PyTorch**, and **NLTK**. Instead of relying on heavy generative models, this assistant uses a custom feedforward neural network to classify user messages into predefined intents from a JSON dataset and trigger dynamic Python functions.

---

## ✨ Features

* **Custom PyTorch Neural Network:** A 3-layer fully connected neural network (`128 -> 64 -> output`) with ReLU activations and Dropout (`0.5`) to prevent overfitting.
* **NLP Preprocessing Pipeline:** Uses NLTK for tokenization, WordNet lemmatization, and Bag-of-Words (BoW) vectorization.
* **JSON-Driven Intents:** Easily add new conversational topics, training patterns, and responses in `intents.json` without touching the core model architecture.
* **Dynamic Function Mapping:** Maps specific intents (like `stocks`) directly to executable Python functions (`get_stocks()`).
* **Deployment Ready:** Ready to run via CLI or wrap into a **Streamlit** web interface, **FastAPI** REST API, and **Docker** container.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **NLP:** NLTK (`nltk`)
* **Data Handling:** NumPy, JSON
* **Optional Web / Deployment:** FastAPI, Uvicorn, Streamlit, Docker

---

## 📁 Project Structure

```text
├── intents.json          # Training patterns, intent tags, and canned responses
├── main.py               # Neural network architecture, NLP pipeline, training & CLI loop
├── chatbot_model.pth     # Saved PyTorch model weights (generated after training)
├── dimensions.json       # Saved input/output layer dimensions (generated after training)
├── requirements.txt      # Python dependencies
├── app.py                # (Optional) FastAPI backend server
├── frontend.py           # (Optional) Streamlit chat interface
└── Dockerfile            # (Optional) Container configuration for deployment
```

---

## 🚀 Getting Started

### 1. Install Dependencies
Clone the repository and install the required packages:

```bash
pip install torch numpy nltk
```
*(Optional: If using the web UI or API, also run `pip install streamlit fastapi uvicorn`)*

### 2. Download NLTK Resources
Run Python once to download the required tokenizer and lemmatizer dictionaries:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

### 3. Train the Model
Before chatting, you must train the neural network on `intents.json` to generate `chatbot_model.pth` and `dimensions.json`.

In `main.py`, ensure the training block inside `if __name__ == '__main__':` is uncommented:

```python
if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    assistant.save_model('chatbot_model.pth', 'dimensions.json')
```

Run the training script:
```bash
python main.py
```

### 4. Run the Chatbot (CLI)
Once trained, switch `main.py` back to inference mode to load the saved weights and chat in your terminal:

```python
if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')
        if message == '/quit':
            break
        print(assistant.process_message(message))
```

Start chatting:
```bash
python main.py
```
*(Type `/quit` to exit the chat loop.)*

---

## 🌐 Running with Streamlit or FastAPI (Optional)

* **Streamlit Frontend:**
  ```bash
  streamlit run frontend.py
  ```
* **FastAPI Backend:**
  ```bash
  uvicorn app:app --reload
  ```
  Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

## 🧩 Customizing Intents & Functions

1. **Add New Intents:** Open `intents.json` and append a new object to the `"intents"` list:
   ```json
   {
     "tag": "weather",
     "patterns": ["What is the weather?", "Is it raining?", "Weather forecast"],
     "responses": ["Checking the local forecast for you!"]
   }
   ```
2. **Map a Python Function:** Pass a dictionary mapping the `"tag"` name to a Python function when initializing `ChatbotAssistant`:
   ```python
   assistant = ChatbotAssistant('intents.json', function_mappings={'weather': get_weather})
   ```
3. **Retrain:** Re-run `train_model()` and `save_model()` whenever you modify `intents.json` so the vocabulary and output layer update accordingly.

---

## 🔮 Future Improvements

* **Confidence Thresholding:** Add a `torch.softmax` probability check in `process_message()` so the bot replies with *"I'm not sure I understand"* when confidence falls below a threshold (e.g., `0.75`).
* **Dynamic Response Formatting:** Return values directly from mapped functions (such as `get_stocks()`) into the chat string rather than printing to stdout.