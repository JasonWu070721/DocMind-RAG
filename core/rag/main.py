from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.llm.local_llm import get_llm
from core.vectorstore.chroma_store import get_vectorstore

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def answer(query: str) -> str:
    # Vector store & retriever
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = get_llm()

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "Use the provided context to answer the question. "
                "If the answer is not in the context, say you don't know.\n\n"
                "Context:\n{context}"
            ),
            ("human", "{input}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)


if __name__ == "__main__":
    print(answer("block_content"))
