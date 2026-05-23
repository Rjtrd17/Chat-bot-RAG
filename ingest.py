import os
import shutil

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")

print("Reading from:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise Exception(f"Data folder not found: {DATA_PATH}")

print("Files found:", os.listdir(DATA_PATH))


# ===============================
# SAFE LOADER
# ===============================
def safe_load(loader):
    docs = []
    try:
        for doc in loader.load():
            docs.append(doc)
    except Exception as e:
        print(f"Loader failed: {e}")
    return docs


# ===============================
# LOAD DOCUMENTS
# ===============================
documents = []

# PDF
documents += safe_load(
    DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
)

# TXT
documents += safe_load(
    DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
)

# CSV
documents += safe_load(
    DirectoryLoader(
        DATA_PATH,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        show_progress=True
    )
)

# Excel
documents += safe_load(
    DirectoryLoader(
        DATA_PATH,
        glob="**/*.xls*",
        loader_cls=UnstructuredExcelLoader,
        show_progress=True
    )
)

# SQL
documents += safe_load(
    DirectoryLoader(
        DATA_PATH,
        glob="**/*.sql",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
)


# ===============================
# VALIDATION
# ===============================
print(f"\nLoaded {len(documents)} documents")

if len(documents) == 0:
    raise Exception("No files loaded. Check your data folder and file extensions.")


# ===============================
# SPLIT DOCUMENTS
# ===============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

if len(chunks) == 0:
    raise Exception("No chunks created.")


# ===============================
# EMBEDDINGS
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ===============================
# CLEAR OLD VECTOR STORE
# ===============================
if os.path.exists(VECTOR_PATH):
    print("Clearing old vector store...")
    shutil.rmtree(VECTOR_PATH)


# ===============================
# CREATE CHROMA DB
# ===============================
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_PATH
)

vectordb.persist()

print("\nVector store created successfully!")
print(f"Saved to: {VECTOR_PATH}")