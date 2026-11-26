import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

sys_prompt_orq = SystemMessage(
    content=(
        "Analiza la intención de la consulta del usuario y clasifícala en una de estas dos categorías: "
        "'Documental' (búsqueda de información o datos específicos) o 'Analista' (interpretación, comparación o razonamiento complejo). "
        "Responde única y exclusivamente con la palabra 'Documental' o 'Analista', sin explicaciones adicionales."
    )
)

sys_prompt_analista = SystemMessage(
    content=(
        "Eres un especialista analista. Examina el caso basándote únicamente en los datos proporcionados y responde con: "
        "1) problema identificado, 2) información faltante, y 3) dos recomendaciones. Está estrictamente prohibido inventar datos, "
        "hacer suposiciones o asumir acciones realizadas."
    )
)

sys_prompt_documental = SystemMessage(
    content=(
        "Actúa como un experto en extracción documental. Responde basándote exclusivamente en el contenido literal del texto. "
        "Está estrictamente prohibido hacer suposiciones o extraer conclusiones. "
        "Si un dato no aparece, indica: 'El documento no proporciona esta información.'"
    )
)


class Estado(TypedDict, total=False):
  prompt_usuario: str
  documento: str
  categoria: str
  respuesta_final: str


def orquestador(estado: Estado):
  usuario = estado["prompt_usuario"]
  solicitud_orquestador = HumanMessage(content=usuario)

  respuesta = llm.invoke([sys_prompt_orq, solicitud_orquestador])
  seleccion = respuesta.content.strip()
  if seleccion not in ["Documental", "Analista"]:
    seleccion = "Analista"

  return {"categoria": seleccion}


def nodo_analista(estado: Estado):
  doc = estado.get("documento", "No se proporcionó ningún documento.")
  contenido = f"Documento:\n{doc}\n\nConsulta:\n{estado['prompt_usuario']}"
  respuesta = llm.invoke([sys_prompt_analista, HumanMessage(content=contenido)])
  return {"respuesta_final": respuesta.content}


def nodo_documental(estado: Estado):
  doc = estado.get("documento", "No se proporcionó ningún documento.")
  contenido = f"Documento:\n{doc}\n\nConsulta:\n{estado['prompt_usuario']}"
  respuesta = llm.invoke(
      [sys_prompt_documental, HumanMessage(content=contenido)]
  )
  return {"respuesta_final": respuesta.content}


def decidir_ruta(estado: Estado):
  return estado["categoria"].lower()


builder = StateGraph(Estado)

builder.add_node("orquestador", orquestador)
builder.add_node("analista", nodo_analista)
builder.add_node("documental", nodo_documental)

builder.add_edge(START, "orquestador")
builder.add_conditional_edges(
    "orquestador",
    decidir_ruta,
    {"analista": "analista", "documental": "documental"},
)
builder.add_edge("analista", END)
builder.add_edge("documental", END)

agente = builder.compile()

texto_documento = "Aquí va el contenido del documento a consultar o analizar."

while True:
  usuario = input("User: ").strip()
  if usuario.lower() == "salir":
    break
  if not usuario:
    continue

  resultado = agente.invoke(
      {"prompt_usuario": usuario, "documento": texto_documento}
  )
  print(f"\n[Ruta elegida]: {resultado['categoria']}")
  print(f"[Respuesta]: {resultado['respuesta_final']}\n")
