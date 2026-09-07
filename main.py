from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load pdf
reader = PdfReader("sop_kebijakan_klaim_garansi.pdf")
docs = [
    Document(page_content=page.extract_text(), metadata={"page": i + 1})
    for i, page in enumerate(reader.pages)
]

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=500
)
splits = text_splitter.split_documents(docs)

# Vectorization and database vector (chroma)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialization LLM model and prompt
llm = ChatOllama(model="llama3", temperature=0)

template = """You are an official Customer Experience & Technical Support AI Assistant for Electropedia.
Your task is to answer user inquiries strictly based on the provided SOP context.

CRITICAL RULES:
1. OUTPUT LANGUAGE: ALWAYS respond strictly in ENGLISH.
2. DIRECT RESPONSE: Answer the question directly. Do NOT explain RMA processes, return steps, or ask for documents if the claim is VOID / REJECTED.

DEFINITION OF "VOID / EXCLUSION / TOLERANCE":
- "Void", "Factory Tolerance", "Customer Induced Damage (CID)", or "Exclusion" means the claim is REJECTED / NOT ELIGIBLE.
- If a condition is VOID, state clearly that the user CANNOT claim the warranty and NO RMA will be issued.
- NEVER list return steps or RMA timelines for VOID claims.

MATHEMATICAL LOGIC FOR DEAD PIXELS (ELC-PC):
- 1, 2, 3, 4, 5 center dots = Factory Tolerance / Void -> CLAIM REJECTED / INELIGIBLE.
- More than 5 center dots (6, 7, 8+) = Covered -> CLAIM ELIGIBLE (Repair/Swap).

MATHEMATICAL LOGIC FOR BATTERY HEALTH (ELC-RF):
- If the battery health above 70%, the customer cannot claim the warranty, but if the battery health is under 70%, the customer can claim the warranty


Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Make chain RAG modern (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# chat loop interactive
print("=" * 50)
print("Bot is ready. (Type 'exit' or 'bye' to end chat)")
print("=" * 50)

while True:
    query = input("\nCustomer: ")

    # Cek perintah untuk keluar dari program
    if query.lower() in ["exit", "bye"]:
        print("Bot: See you later.")
        break

    # Ignore if user only press Enter without any text
    if not query.strip():
        continue

    response = rag_chain.invoke(query)
    print(f"\nBot: {response}")