import os
import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough, RunnableLambda
#from langchain_core.output_parsers import StrOutputParser

# config page design
st.set_page_config(page_title="Electropedia CS Bot", page_icon="⚡")
st.title("⚡ Electropedia Customer Support")
st.caption("Official AI Assistant for Warranty & Claims")

DB_PATH = "./chroma_db"

#load vector database

@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    if os.path.exists(DB_PATH):
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

    reader = PdfReader(
        "sop_kebijakan_klaim_garansi.pdf"
    )

    docs = []

    for i, page in enumerate(reader.pages):
        docs.append(
            Document(
                page_content=page.extract_text(),
                metadata={"page": i + 1}
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
)

#LLM
llm = ChatOllama(
    model="llama3",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
"""
You are an official Customer Support Assistant.

RULES:

- Answer ONLY using SOP context.
- Never use external knowledge.
- Never make assumptions.
- Always answer in English.
- If the SOP does not contain the answer, reply:

"According to the SOP, this information is not specified."

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)

#chat memory
if "message" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am Electropedia CS Assistant. How can I help you today?"
        }
    ]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

#user input

if user_input := st.chat_input("Type your question here..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Searching SOP..."):

            # retrieval
            docs = retriever.invoke(user_input)

            if len(docs) == 0:

                response = (
                    "According to the SOP, this information is not specified."
                )

                st.write(response)

            else:

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                history = "\n".join(
                    [
                        f"{m['role']}: {m['content']}"
                        for m in st.session_state.messages[-6:]
                    ]
                )

                response = chain.invoke(
                    {
                        "chat_history": history,
                        "context": context,
                        "question": user_input
                    }
                )

                st.write(response)

                # source pages
                pages = sorted(
                    set(
                        doc.metadata["page"]
                        for doc in docs
                    )
                )

                st.caption(
                    f"Sources: Page {', '.join(map(str, pages))}"
                )

                # debug panel
                with st.expander("Retrieved Context"):

                    for doc in docs:

                        st.markdown(
                            f"### Page {doc.metadata['page']}"
                        )

                        st.write(
                            doc.page_content[:500]
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )