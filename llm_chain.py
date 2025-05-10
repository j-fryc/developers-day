import abc

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from knowledgebase import KnowledgeBase


class MyLLMChain:
    @abc.abstractmethod
    def run(self, question: str) -> str:
        """Run the LLM chain with the given question."""
        pass


class RAGLLMChain(MyLLMChain):
    def __init__(
            self,
            openai_api_key: str,
            knowledge_base: KnowledgeBase,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.7,
    ):
        self.knowledge_base = knowledge_base
        retriever = RunnableLambda(
            lambda x: {"context": "\n\n".join(
                doc.page_content for doc in self.knowledge_base.search(x["question"], k=3)
            ), "question": x["question"]}
        )

        self.prompt_template = PromptTemplate.from_template("""Udziel odpowiedzi na pytanie na podstawie kontekstu.
            Kontekst: {context}
            Pytanie: {question}""")

        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)

        self.chain = (
                retriever
                | self.prompt_template
                | self.llm
                | StrOutputParser()
        )

    def run(self, question: str) -> str:
        return self.chain.invoke({"question": question})
