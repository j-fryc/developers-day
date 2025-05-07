import streamlit as st

from knowledgebase import KnowledgeBase
from llm_chain import RAGLLMChain


class StreamlitApp:
    def __init__(
            self,
            knowledge_base: KnowledgeBase,
            llm_chain: RAGLLMChain,
    ):
        self.knowledge_base = knowledge_base
        self.llm_chain = llm_chain

        st.set_page_config(page_title="Developers Day Chat Bot AI", page_icon="🤖")
        st.title("🤖 Developers Day Chat Bot AI")
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def run(self) -> None:
        self.handle_file_upload()
        self.display_messages()
        self.handle_user_input()

    def handle_file_upload(self) -> None:
        uploaded_file = st.file_uploader("Dodaj plik PDF", type="pdf")
        if uploaded_file:
            if self.knowledge_base.process_pdf_to_vectorstore(uploaded_file):
                st.success("📄 Dokument przetworzony i dodany do bazy wiedzy.")
            else:
                st.error("Wystąpił błąd podczas przetwarzania pliku PDF.")

    def display_messages(self) -> None:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def handle_user_input(self) -> None:
        prompt: str | None = st.chat_input("Napisz wiadomość...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("AI pisze..."):
                    reply: str = self.llm_chain.run(prompt)

                    st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
