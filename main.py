import streamlit as st
import requests
import json
from time import sleep

def get_ollama_response(prompt, system_prompt, model="phi"):
    """Get response from Ollama API with timeout and chunking"""
    try:
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nLeBron James:"
        
        # Add a status indicator
        with st.spinner('LeBron is thinking...'):
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "options": {
                        "num_ctx": 512,
                        "num_predict": 256,
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                    }
                },
                stream=True)
            
            # Initialize complete response
            complete_response = ""
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        complete_response += json_response['response']
            
            return complete_response

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI setup with memory management
st.set_page_config(page_title="LeBron James Chat", layout="wide")

# LeBron James personality prompt - shortened for efficiency
LEBRON_PROMPT = """You are LeBron James, NBA legend. Keep responses under 100 words. Focus on:
- Your NBA championships and MVP awards
- Your journey from Akron, Ohio
- Your business ventures and social initiatives
- Your playing style and basketball knowledge
Speak confidently but humble, like LeBron."""

# Streamlit UI setup
st.title("Chat with LeBron James 👑🏀")
st.sidebar.image("https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/1966.png&w=350&h=254", caption="LeBron James")

# Initialize chat history with a limit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Limit chat history to last 10 messages for performance
MAX_MESSAGES = 10

# Display chat history
for message in st.session_state.messages[-MAX_MESSAGES:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask LeBron something"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to message history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display LeBron's response
    with st.chat_message("assistant"):
        response = get_ollama_response(prompt, LEBRON_PROMPT)
        st.markdown(response)
    
    # Add to message history and maintain limit
    st.session_state.messages.append({"role": "assistant", "content": response})
    if len(st.session_state.messages) > MAX_MESSAGES * 2:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES * 2:]

# Sidebar with minimal content
with st.sidebar:
    st.markdown("""
    ## Quick Stats
    - 4× NBA champion
    - 4× MVP
    - Born in Akron, OH
    """)

    # Model selector
    model = st.selectbox(
        "Select Model",
        ["phi", "mistral:7b-instruct-q4", "llama2:4bit"],
        index=0
    )

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Memory usage warning
    st.info('💡 If the app feels slow, try clearing the chat or selecting a smaller model.')
