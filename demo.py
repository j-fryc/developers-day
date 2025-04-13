import httpx
import streamlit as st
from openai import OpenAI

client = OpenAI()

st.set_page_config(page_title="Prosty Chat z AI", page_icon="🤖")

st.title("🤖 Developers Day Chat Bot AI")

# Inicjalizacja sesji
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie historii wiadomości
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Wprowadzanie nowej wiadomości
if prompt := st.chat_input("Napisz wiadomość..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI pisze..."):
            response = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
