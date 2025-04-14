import streamlit as st
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
import tempfile
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


client = OpenAI(
    api_key=st.secrets.get("OPENAI_API_KEY")
)
embedding_model = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

PG_CONN_STRING = st.secrets["PGVECTOR_URL"]
COLLECTION_NAME = "chatbot_docs"

st.set_page_config(page_title="Prosty Chat z AI", page_icon="🤖")

st.title("🤖 Developers Day Chat Bot AI")

# Inicjalizacja sesji
if "messages" not in st.session_state:
    st.session_state.messages = []

vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=PG_CONN_STRING,
    embedding_function=embedding_model
)



# Funkcja przetwarzająca PDF i zapisująca embeddingi do pgvector
def process_pdf_to_pgvector(file):
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

    vectorstore.add_documents(documents)
    st.success("📄 Dokument przetworzony i dodany do bazy wiedzy.")

# Upload PDF i przetwarzanie
uploaded_file = st.file_uploader("Dodaj plik PDF", type="pdf")
if uploaded_file:
    process_pdf_to_pgvector(uploaded_file)



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
            docs = vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            full_prompt = f"{context}\n\nUżytkownik pyta: {prompt}"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Odpowiadaj na podstawie dostarczonego kontekstu."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
