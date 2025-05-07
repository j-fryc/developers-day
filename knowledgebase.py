import abc
import os
import tempfile
from typing import List

import PyPDF2
# import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


class KnowledgeBase(abc.ABC):
    @abc.abstractmethod
    def process_pdf_to_vectorstore(self, file) -> bool:
        """Process PDF file and add its content to the vectorstore."""
        pass

    @abc.abstractmethod
    def search(self, query: str, k: int = 3) -> list[Document]:
        """Search the vectorstore for relevant documents based on the query."""
        pass


class InMemoryKnowledgeBase(KnowledgeBase):
    def __init__(self, openai_api_key):
        self.vectorstore = InMemoryVectorStore(
            OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_pdf_to_vectorstore(self, file) -> bool:
        try:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()

            chunks = self.text_splitter.split_text(full_text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error processing PDF: {e}")  # Log the error for debugging
            return False

    def search(self, query: str, k: int = 3) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)


class PGVectorKnowledgeBase(KnowledgeBase):
    def __init__(self, pg_conn_string: str, collection_name: str, openai_api_key: str):
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=pg_conn_string,
            embedding_function=self.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_pdf_to_vectorstore(self, file) -> bool:
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

    def search(self, query: str, k: int = 3) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)
