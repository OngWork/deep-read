from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import S3FileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
import boto3
import json
import tempfile

# configure AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

embeddings = PineconeEmbeddings(
    model="multilingual-e5-large", 
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deepread-frontend-url.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="meta.llama3-8b-instruct-v1:0",
    model_kwargs={"temperature": 0.6}
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        s3_client.upload_fileobj(file.file, BUCKET_NAME, file.filename)
        return {"status": "Success", "filename": file.filename}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/process-s3-file")
async def process_s3_file(filename: str):
    try:
        # 1. create a temporary file to store the downloaded PDF 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # 2. download the file from S3 to the temporary location
            s3_client.download_fileobj(BUCKET_NAME, filename, tmp_file)
            tmp_path = tmp_file.name

        # 3. use PyPDFLoader to load the PDF 
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 4. split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # 5. Send to Pinecone (Integrated Embeddings)
        PineconeVectorStore.from_documents(
            documents=splits,
            index_name="deepread-index",
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )

        os.remove(tmp_path)
        
        print(f"Indexed {filename} to Pinecone successfully (Using PyPDF)")
        return {"status": "Success"}
    except Exception as e:
        print(f"Error in process: {str(e)}")
        return {"status": "Error", "message": str(e)}

@app.post("/chat")
async def chat(query: str = Form(...), history: str = Form("[]")):
    try:
        print(f"--- New Chat Request ---")
        print(f"Query: {query}")
        
        # 1. history
        chat_history = json.loads(history)
        
        # 2. Connect to Pinecone
        print("Connecting to Pinecone...")
        vectorstore = PineconeVectorStore(
            index_name="deepread-index", 
            embedding=embeddings, 
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # 3. Find relevant documents from Pinecone
        print("Searching for context...")
        docs = vectorstore.similarity_search(query, k=4)
        print(f"Found {len(docs)} relevant documents")
        
        if not docs:
            return {"answer": "There is no relevant information available.", "sources": []}

        # 4. Call LLM
        print("Invoking Llama 3 on Bedrock...") 
        context_text = "\n\n".join([doc.page_content for doc in docs])
        full_prompt = f"History: {chat_history}\n\nContext: {context_text}\n\nQuestion: {query}"
        
        response = llm.invoke(full_prompt)
        print("LLM Response received")
        
        sources = []
        for doc in docs:
            source_info = {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1
            }
            if source_info not in sources:
                sources.append(source_info)
                
        answer_text = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": response.content if hasattr(response, 'content') else str(response),
            "sources": sources
        }
    except Exception as e:
        print(f"ERROR in /chat: {str(e)}") 
        return {"status": "Error", "message": str(e)}