
import streamlit as st
import nltk
from main import ChatbotAssistant, get_stocks

# Download NLTK data required for tokenization and lemmatization
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Cache the model initialization so it only loads once per session
@st.cache_resource
def load_chatbot():
    # Initialize the assistant with the same parameters as main.py
    assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
    assistant.parse_intents()
    # Load the trained weights and dimensions
    assistant.load_model('chatbot_model.pth', 'dimensions.json')
    return assistant

assistant = load_chatbot()

# Configure the Streamlit page
st.title("🤖 PyTorch Chatbot")
st.caption("A neural network-based intent classifier")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the message through the PyTorch model
    response = assistant.process_message(prompt)
    
    # Handle cases where the model returns None
    if not response:
        response = "I'm sorry, I don't understand that."

    # Add assistant response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)