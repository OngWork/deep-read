from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# อนุญาตให้ React ติดต่อกับ FastAPI ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

persist_directory = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Load and Split
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter)
    
    # Store in ChromaDB
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    os.remove(file_path)
    return {"status": "Success", "filename": file.filename}

@app.post("/chat")
async def chat(query: str = Form(...)):
    # Load Vector DB
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Setup RAG Chain
    llm = ChatOllama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    
    result = qa_chain.invoke(query)
    return {"answer": result["result"]}