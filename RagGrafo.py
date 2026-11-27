import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

# 1. Configuración de entorno y modelo LLM
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

# 2. Carga, fragmentación e indexación vectorial del documento
loader = PyPDFLoader(str(BASE / "data" / "Anillos.pdf"))
documentos = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documentos)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

base_vectorial = InMemoryVectorStore(embeddings)
base_vectorial.add_documents(chunks)

retriever = base_vectorial.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# 3. Únicos 3 Metaprompts del sistema
sys_prompt_orq = SystemMessage(
    content=(
        "Eres un enrutador semántico estricto. Analiza la consulta del usuario (y el contexto "
        "documental si se proporciona) para decidir la ruta:\n"
        "- 'RAG': ÚNICAMENTE si la pregunta está estrictamente relacionada con el contenido de la "
        "base documental (Anillos.pdf) y puede responderse con dicha información.\n"
        "- 'CHAT': Para saludos, conversación general o cualquier consulta que NO pertenezca "
        "estrictamente a la base documental.\n\n"
        "Ante la duda o falta de relación directa con el documento, responde 'CHAT'. "
        "Responde única y exclusivamente con la palabra 'RAG' o 'CHAT', sin explicaciones."
    )
)

sys_prompt_chat = SystemMessage(
    content=(
        "Eres un asistente virtual claro y preciso. "
        "Si el mensaje incluye 'Contexto_Recuperado_Base', actúa como especialista documental y "
        "responde únicamente con la información literal de ese texto sin inventar datos. "
        "Si no se incluye contexto documental, responde de forma conversacional y natural."
    )
)

system_reinterpretacion = (
    "Re interpreta la solicitud del usuario y escribela brevemente de forma clara y sencilla "
    "usando palabras tecnicas si el usuario las utiliza"
)


# 4. Definición del Estado del Grafo
class Estado(TypedDict, total=False):
  prompt_usuario: str
  categoria: str
  respuesta_final: str


# 5. Definición de Nodos
def orquestador(estado: Estado):
  usuario = estado["prompt_usuario"]
  respuesta = llm.invoke([sys_prompt_orq, HumanMessage(content=usuario)])
  seleccion = respuesta.content.strip().upper()

  if seleccion not in ["CHAT", "RAG"]:
    seleccion = "CHAT"

  return {"categoria": seleccion}


def nodo_chat(estado: Estado):
  solicitud = HumanMessage(content=estado["prompt_usuario"])
  respuesta = llm.invoke([sys_prompt_chat, solicitud])
  return {"respuesta_final": respuesta.content}


def nodo_rag(estado: Estado):
  usuario = estado["prompt_usuario"]

  # Reinterpretación y búsqueda semántica
  reinterpretacion = llm.invoke([system_reinterpretacion, usuario])
  resultados = retriever.invoke(reinterpretacion.content)

  contexto = ""
  for documento in resultados:
    contexto += documento.page_content + "\n\n"

  # Validación estricta de pertenencia a la base documental con sys_prompt_orq
  validacion = llm.invoke([
      sys_prompt_orq,
      HumanMessage(
          content=(
              f"Contexto_Recuperado_Base:\n{contexto}\n\n"
              f"Pregunta_Usuario:\n{usuario}"
          )
      ),
  ])

  if validacion.content.strip().upper() != "RAG":
    return Command(
        update={"categoria": "CHAT (Devuelto desde RAG)"},
        goto="chat",
    )

  solicitud = HumanMessage(
      content=(
          f"Contexto_Recuperado_Base:\n{contexto}\n\n"
          f"Pregunta_Usuario:\n{usuario}"
      )
  )
  respuesta = llm.invoke([sys_prompt_chat, solicitud])
  return {"respuesta_final": respuesta.content}


# 6. Función de Enrutamiento Condicional
def decidir_ruta(estado: Estado):
  return estado["categoria"]


# 7. Construcción y Compilación del Grafo
builder = StateGraph(Estado)

builder.add_node("orquestador", orquestador)
builder.add_node("chat", nodo_chat)
builder.add_node("rag", nodo_rag)

builder.add_edge(START, "orquestador")
builder.add_conditional_edges(
    "orquestador",
    decidir_ruta,
    {
        "CHAT": "chat",
        "RAG": "rag",
    },
)
builder.add_edge("chat", END)
builder.add_edge("rag", END)

agente = builder.compile()

# 8. Generar imagen del grafo
png_bytes = agente.get_graph().draw_mermaid_png()
with open(BASE / "RagGrafo.png", "wb") as f:
  f.write(png_bytes)

# 9. Chat usuario
while True:
    usuario = input("Usuario: ").strip()
    if usuario.lower() == "salir":
      break
    if not usuario:
      continue

    resultado = agente.invoke({"prompt_usuario": usuario})
    print(f"\n[Ruta elegida]: {resultado['categoria']}")
    print(f"[Respuesta]: {resultado['respuesta_final']}\n")