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


# 3. Dividir el documento en chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documentos)

# 4. Modelo de embeddings.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# 5. Crear el almacén vectorial en memoria
base_vectorial = InMemoryVectorStore(embeddings)
#base_vectorial.reset_collection()
base_vectorial.add_documents(chunks)


# 6. Crear el recuperador.
retriever = base_vectorial.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# 7. Metaprompt e inicialización de memoria.
system_Documental = """
    Eres el especialista documental.
    Responde únicamente con la información del texto proporcionado.
    Si el texto no contiene la respuesta, indícalo. No inventes datos.
"""

system_Reescritor = """
    Eres un experto en optimización de consultas para búsqueda documental.
    Reestructura y reescribe la solicitud del usuario para que sea clara, precisa y directa.
    Devuelve únicamente la pregunta reescrita, sin explicaciones adicionales.
"""

# 8. Ejecución del chat.
while True:

    print("Usuario: ", end="", flush=True)
    usuario = input().strip()

    if usuario.lower() == "salir":
        break

    if not usuario:
        continue

    #Reestructurar y reescribir la solicitud del usuario.
    instruccion_reescritura = SystemMessage(content=system_Reescritor)

    pregunta_reescrita = llm.invoke([
        instruccion_reescritura,
        HumanMessage(content=usuario)
    ]).content.strip()

    # Recuperar los documentos más similares con la pregunta reescrita.
    resultados = retriever.invoke(pregunta_reescrita)

    # Construir el contexto.
    contexto = ""

    for documento in resultados:
        contexto += documento.page_content + "\n\n"

    # Construir los mensajes.
    instruccion_RAG = SystemMessage(content=system_Documental)

    solicitud = HumanMessage(
        content=f"""
        Contexto_Recuperado_Base: {contexto}
        Pregunta_Usuario: {pregunta_reescrita}
        """
    )

    # Invoke 2: Generar la respuesta final basada en el contexto.
    respuesta = llm.invoke([
        instruccion_RAG,
        solicitud
    ])

    print("Respuesta:", respuesta.content, "\n")
