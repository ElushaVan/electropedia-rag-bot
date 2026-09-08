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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300
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
    1. OUTPUT LANGUAGE: ALWAYS respond strictly in ENGLISH.
    2. FINAL ANSWER ONLY: Output ONLY the direct, polished response meant for the customer. NEVER display "Thinking Step", "User Inquiry", internal reasoning tags, or category tags in your final message.
    3. MATCH ISSUE DIRECTLY: Address ONLY the user's specific problem using the correct product category section from the context.
    4. ABSENCE OF DATA: If the specific issue or condition is not mentioned in the context, state that the information is not specified in the SOP.

    DEFINITION OF "VOID / EXCLUSION / TOLERANCE":
    - "Void", "Factory Tolerance", "Customer Induced Damage (CID)", or "Exclusion" means the claim is REJECTED / NOT ELIGIBLE.
    - If a condition is VOID, state clearly that the user CANNOT claim the warranty and NO RMA will be issued.
    - NEVER list return steps or RMA timelines for VOID claims.

    MATHEMATICAL LOGIC FOR DEAD PIXELS (ELC-PC):
    - 1, 2, 3, 4, 5 center dots = Factory Tolerance / Void -> CLAIM REJECTED / INELIGIBLE.
    - More than 5 center dots (6, 7, 8+) = Covered -> CLAIM ELIGIBLE (Repair/Swap).

    REASONING FORMAT (Follow this structure):
    User Inquiry: [User Question]
    Thinking Step:
    - Product Category: [e.g., ELC-SM]
    - User Issue: [e.g., Bootloop]
    - Matching Rule in Context: [e.g., System failure covered under 12-month warranty]
    Answer: [Direct response without fluff]
    

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
