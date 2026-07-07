from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Document Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Globals (simple in-memory cache) ─────────────────────────────────────────

_embeddings = None
_chain      = None
_retriever  = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def build_chain():
    global _chain, _retriever
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings()
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the context below to answer. If unsure, say "Not found in my notes."

Context: {context}

Question: {question}
Answer:"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    _chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """Upload and index PDF/TXT files."""
    all_chunks = []
    splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for f in files:
        suffix = ".pdf" if f.filename.endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
        docs   = loader.load()
        all_chunks.extend(splitter.split_documents(docs))
        os.unlink(tmp_path)

    Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        persist_directory="./chroma_db"
    )

    # Rebuild chain with updated vectorstore
    build_chain()

    return {"status": "ok", "chunks": len(all_chunks)}


class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskRequest):
    """Answer a question using the RAG chain."""
    global _chain, _retriever

    # Auto-load chain if chroma_db already exists
    if _chain is None:
        if not os.path.exists("./chroma_db"):
            raise HTTPException(status_code=400, detail="No documents ingested yet.")
        build_chain()

    answer  = _chain.invoke(body.question)
    sources = _retriever.invoke(body.question)

    source_list = []
    for i, doc in enumerate(sources, 1):
        source_list.append({
            "chunk": i,
            "page":  doc.metadata.get("page", "?"),
            "text":  doc.page_content[:200] + "..."
        })

    return {"answer": answer, "sources": source_list}


@app.post("/clear")
def clear_cache():
    """Clear the in-memory chain so it rebuilds on next ask."""
    global _chain, _retriever, _embeddings
    _chain      = None
    _retriever  = None
    _embeddings = None
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)