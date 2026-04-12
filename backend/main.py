from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

app = FastAPI()

# Allow React to connect to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

persist_directory = "./chroma_db"
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"  
)

llm = ChatOllama(
    model="llama3",
    base_url="http://ollama:11434"  
)

# --- 1. Prepare Prompt Template ---
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter)
    
    # Store in Chroma db
    Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    os.remove(file_path)
    return {"status": "Success", "filename": file.filename}

@app.post("/chat")
async def chat(query: str = Form(...)):
    # Retrieve from Chroma db
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever()
    
    # --- 2. Create Chain with LCEL (The First Method) ---
    # We use the | symbol to chain them together, similar to a pipe in Linux/C++ [cite: 11]
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Run the processing system
    response = rag_chain.invoke(query)
    
    return {"answer": response}