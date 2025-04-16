import streamlit as st
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
import tempfile
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

class KnowledgeBase:
    def __init__(self, pg_conn_string, collection_name, openai_api_key):
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=pg_conn_string,
            embedding_function=self.embedding_model
        )

    def process_pdf_to_vectorstore(self, file):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        reader = PyPDF2.PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        self.vectorstore.add_documents(documents)
        return True  # Indicate successful processing


    def search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)

class LLMChainWrapper:  # Renamed for clarity
    def __init__(self, openai_api_key, knowledge_base, model_name="gpt-3.5-turbo", temperature=0.7):
        self.llm = LangchainOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
        self.knowledge_base = knowledge_base  # Store the knowledge base

        self.prompt_template = PromptTemplate.from_template("""Udziel odpowiedzi na pytanie na podstawie kontekstu.
            Kontekst: {context}
            Pytanie: {question}
            """)

        # Define the chain using the pipe operator
        self.chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in self.knowledge_base.search(x["question"], k=3)]), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def run(self, question): #Now only takes the question
        return self.chain.invoke({"question": question})


class StreamlitApp:
    def __init__(self, knowledge_base, llm_chain_wrapper, openai_api_key): # Pass openai_api_key
        self.knowledge_base = knowledge_base
        self.llm_chain_wrapper = llm_chain_wrapper
        self.openai_api_key = openai_api_key # Store the API key
        self.openai_client = OpenAI(api_key=self.openai_api_key)  # Initialize OpenAI client here


        st.set_page_config(page_title="Prosty Chat z AI", page_icon="🤖")
        st.title("🤖 Developers Day Chat Bot AI")
        self.initialize_session_state()

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def run(self):
        self.handle_file_upload()
        self.display_messages()
        self.handle_user_input()

    def handle_file_upload(self):
        uploaded_file = st.file_uploader("Dodaj plik PDF", type="pdf")
        if uploaded_file:
            if self.knowledge_base.process_pdf_to_vectorstore(uploaded_file):
                st.success("📄 Dokument przetworzony i dodany do bazy wiedzy.")
            else:
                st.error("Wystąpił błąd podczas przetwarzania pliku PDF.")

    def display_messages(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def handle_user_input(self):
        if prompt := st.chat_input("Napisz wiadomość..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("AI pisze..."):
                    reply = self.llm_chain_wrapper.run(prompt) # Now pass only the question

                    st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Main execution ---
if __name__ == "__main__":
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    pg_conn_string = st.secrets["PGVECTOR_URL"]
    collection_name = "chatbot_docs"

    knowledge_base = KnowledgeBase(pg_conn_string, collection_name, openai_api_key)
    llm_chain_wrapper = LLMChainWrapper(openai_api_key, knowledge_base)  # Pass KB

    app = StreamlitApp(knowledge_base, llm_chain_wrapper, openai_api_key) # Pass the API key
    app.run()
