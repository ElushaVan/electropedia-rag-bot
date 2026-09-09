import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# config page design
st.set_page_config(page_title="Electropedia CS Bot", page_icon="⚡")
st.title("⚡ Electropedia Customer Support")
st.caption("Official AI Assistant for Warranty & Claims")

# decorator cache_resources so that the RAG engine isnt reloaded everytime it is clicked
@st.cache_resource
def load_rag_chain():
    reader = PdfReader("sop_kebijakan_klaim_garansi.pdf")
    docs = [
        Document(page_content=page.extract_text(), metadata={"page": i + 1})
        for i, page in enumerate(reader.pages)
    ]

#need to fix the chunk later.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k":2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOllama(model="llama3", temperature=0)

    template = """You are an official Customer Experience AI Assistant for Electropedia.
    Your task is to answer user inquiries strictly based on the provided SOP context.

    CRITICAL RULES:
    1. ALWAYS respond in English.
    2. Never reveal internal reasoning.
    3. Never output:
       - Thinking Step
       - User Inquiry
       - Product Category
       - Matching Rule
       - Analysis
    4. Output ONLY the final answer for the customer.
    5. If information is not found in the SOP, state that it is not specified in the SOP.

    Context:
    {context}

    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    return(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# load RAG engine
rag_chain = load_rag_chain()

# initialization chat memory UI streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Electropedia CS Assistant. How can I help you today?"}
    ]

# show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# input chat from user
if user_input := st.chat_input("Type your question here..."):
    # save and show user's chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # bot answer
    with st.chat_message("assistant"):
        with st.spinner("..."):
            response = rag_chain.invoke(user_input)
            st.write(response)

    # save bot's answer to memory
    st.session_state.messages.append({"role": "assistant", "content": response})
