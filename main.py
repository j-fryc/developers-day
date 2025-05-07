import os

from dotenv import load_dotenv

from knowledgebase import KnowledgeBase, InMemoryKnowledgeBase
from llm_chain import RAGLLMChain

# import streamlit as st
# from streamlit.runtime.uploaded_file_manager import UploadedFile

class CliApp:
    def __init__(self, knowledge_base: KnowledgeBase, llm_chain: RAGLLMChain):
        self.knowledge_base = knowledge_base
        self.llm_chain = llm_chain

    def run(self) -> None:
        while True:
            question: str = input("Zadaj pytanie: ")
            if question.lower() == "exit":
                break
            answer: str = self.llm_chain.run(question)
            print(f"Odpowiedź: {answer}")


if __name__ == "__main__":
    load_dotenv('../.env')
    openai_api_key: str = os.getenv("OPENAI_API_KEY")

    # openai_api_key: str = st.secrets["OPENAI_API_KEY"]
    # pg_conn_string: str = st.secrets["PGVECTOR_URL"]
    # collection_name: str = "chatbot_docs"
    # knowledge_base = KnowledgeBase(pg_conn_string, collection_name, openai_api_key)

    knowledge_base = InMemoryKnowledgeBase(openai_api_key)
    knowledge_base.process_pdf_to_vectorstore(open("data/devday.pdf", "rb"))
    llm_chain = RAGLLMChain(openai_api_key, knowledge_base)
    app = CliApp(knowledge_base, llm_chain)
    # app = StreamlitApp(knowledge_base, llm_chain)
    app.run()
