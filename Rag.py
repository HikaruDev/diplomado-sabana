import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# 1. Conectarnos a OpenRouter.
BASE = Path(__file__).resolve().parent

load_dotenv(BASE / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)


# 2. Cargar el documento.
loader = PyPDFLoader(str(BASE / "data" / "Anillos.pdf"))
documentos = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documentos)

embeddings = HuggingFaceEmbeddings(
    model_name = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
)

base_vectorial = InMemoryVectorStore(embeddings)
base_vectorial.add_documents(chunks)

