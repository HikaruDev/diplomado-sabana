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

print(chunks)

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

instruccion = SystemMessage(content=system_Documental)
Historial = []
Resumen = ""

# 8. Ejecución del chat RAG.
while True:
    solicitud = input("\nUser: ").strip()

    if not solicitud:
        continue
    if solicitud.lower() == "salir":
        break

    # Recuperar los 3 fragmentos más relevantes de Anillos.pdf
    docs_relevantes = retriever.invoke(solicitud)
    contexto_pdf = "\n\n".join(doc.page_content for doc in docs_relevantes)

    # Construir el prompt con el contexto del PDF + resumen + pregunta
    prompt_con_contexto = (
        f"Contexto del documento:\n{contexto_pdf}\n\n"
        f"Resumen de la conversación anterior:\n{Resumen}\n\n"
        f"Pregunta del usuario: {solicitud}"
    )

    pregunta = HumanMessage(content=prompt_con_contexto)
    respuesta = llm.invoke([instruccion, pregunta])

    # 1. Imprimir la respuesta inmediatamente
    print(f"\nAI: {respuesta.content}\n")

    # 2. Actualizar historial y resumen
    Historial.extend([pregunta, respuesta])
    Resumen_1 = llm.invoke([
        SystemMessage(
            content=f"Resume la conversación en máximo 500 caracteres. Conserva la información importante. Resumen anterior: {Resumen}"
        ),
        HumanMessage(content=f"Usuario: {solicitud}\nAsistente: {respuesta.content}"),
        HumanMessage(content="Actualiza el resumen de la conversación.")
    ])
    Resumen = Resumen_1.content