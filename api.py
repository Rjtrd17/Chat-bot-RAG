# api.py — Full FastAPI server with Security
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader # Added for security
from pydantic import BaseModel
from typing import Optional, List
import uvicorn, uuid, time, os
from contextlib import asynccontextmanager

# Import your RAG components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- SECURITY SETUP ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Store valid keys (In production: use a database or environment variables)
VALID_KEYS = {
    os.getenv("API_KEY_1", "dev-key-change-me"),
    os.getenv("API_KEY_2", "another-key")
}

async def verify_api_key(key: str = Depends(API_KEY_HEADER)):
    if key not in VALID_KEYS:
        raise HTTPException(403, "Invalid API key")
    return key

# --- MODERN LIFESPAN HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("Initializing RAG Components with Security Layer...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model="llama3.2", temperature=0.1)
 #  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY") )   
    
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the user's question. \n\n"
        "STRICT RULE: If the answer is not in the context, do NOT make up facts. "
        "Instead, state that the data is not available, then provide 3 helpful "
        "suggestions related to the topic.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print("Secure RAG System Ready!")
    yield

app = FastAPI(title="RAG Chatbot API", version="1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    processing_time_ms: int

# Main chat endpoint - Now PROTECTED by verify_api_key
@app.post("/api/chat", response_model=QueryResponse)
async def chat(req: QueryRequest, key: str = Depends(verify_api_key)):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    start = time.time()
    result = rag_chain.invoke({"input": req.question})
    elapsed = int((time.time() - start) * 1000)
    
    sources = [{
        "file": doc.metadata.get("source", "?"),
        "page": doc.metadata.get("page", "?"),
        "content": doc.page_content[:200]
    } for doc in result["context"]]
    
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        session_id=req.session_id or str(uuid.uuid4()),
        processing_time_ms=elapsed
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama3.2"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)