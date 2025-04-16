import os
import tempfile
from typing import List

import PyPDF2
import streamlit as st
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from streamlit.runtime.uploaded_file_manager import UploadedFile


class KnowledgeBase:
    def __init__(self, pg_conn_string: str, collection_name: str, openai_api_key: str):
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=pg_conn_string,
            embedding_function=self.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_pdf_to_vectorstore(self, file: UploadedFile) -> bool:
        tmp_path: str = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            reader = PyPDF2.PdfReader(tmp_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()

            os.remove(tmp_path)

            chunks = self.text_splitter.split_text(full_text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error processing PDF: {e}")  # Log the error for debugging
            return False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def search(self, query: str, k: int = 3) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)


class LLMChain:
    def __init__(
            self,
            openai_api_key: str,
            knowledge_base: KnowledgeBase,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.7,
    ):
        self.llm = LangchainOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
        self.knowledge_base = knowledge_base

        self.prompt_template = PromptTemplate.from_template("""Udziel odpowiedzi na pytanie na podstawie kontekstu.
            Kontekst:
            Pytanie: {question}""")

        self.chain = (
                {"context":
                     lambda x: "\n\n".join(
                         [doc.page_content for doc in self.knowledge_base.search(x["question"], k=3)]),
                 "question":
                     RunnablePassthrough(),
                 }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
        )

    def run(self, question: str) -> str:
        return self.chain.invoke({"question": question})


class StreamlitApp:
    def __init__(
            self,
            knowledge_base: KnowledgeBase,
            llm_chain: LLMChain,
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
        uploaded_file: UploadedFile | None = st.file_uploader("Dodaj plik PDF", type="pdf")
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


if __name__ == "__main__":
    openai_api_key: str = st.secrets["OPENAI_API_KEY"]
    pg_conn_string: str = st.secrets["PGVECTOR_URL"]
    collection_name: str = "chatbot_docs"

    knowledge_base = KnowledgeBase(pg_conn_string, collection_name, openai_api_key)
    llm_chain = LLMChain(openai_api_key, knowledge_base)

    app = StreamlitApp(knowledge_base, llm_chain)
    app.run()