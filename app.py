import streamlit as st
import requests

st.title("Chatbot powered by MCP")

query = st.text_input("Ask your question:")

if st.button("Send"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"query": query}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(f"**Response:**\n\n{data['response']}")
                else:
                    st.error(f"Error from server: {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
