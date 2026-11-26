import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from PIL import Image
from io import BytesIO

# 1. Conectarnos a OpenRouter usando el mismo .env de los chats.
load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

system_orquestador = """
    Eres un clasificador de agentes especializados. Dada la instrucción del usuario, identifica si quiere
    una respuesta breve o una respuesta compleja, en caso de que quiera una respuesta breve, responde Breve y
    si quiere una respuesta más estructurada responde Analista.
    Breve: Cuando se pida una definición, consulta o algo similar
    Analista: Cundo se pida un análisis, plan cosas así
"""

system_Breve = """
    Responde Brevemente a la solicitud del usuario
"""

system_Analista = """
    Analiza la instrucción del usuario y provee un analisis de los requerimientos
"""

system_Plan = """
    Genera un plan para satisfacer los requerimientos necesarios a la solicitud del usuario
"""

system_Final = """
    Responde de forma estructurada a la solicitud del usuario
"""

class Estado(TypedDict, total=False):
    prompt_usuario: str
    especializado: str
    respuesta_A: str
    respuesta_P: str
    respuesta: str

def orquestador(estado: Estado):
    usuario = estado["prompt_usuario"]
    instruccion_orquestador = SystemMessage(content=system_orquestador)
    solicitud_orquestador = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    seleccion = str(respuesta.content).strip().capitalize()

    if seleccion not in ["Breve", "Analista"]:
        seleccion = "Breve"

    return {"especializado" : seleccion}

def Breve(estado: Estado):
    usuario = estado["prompt_usuario"]
    instruccion_Breve = SystemMessage(content=system_Breve)
    solicitud_Breve = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_Breve,
        solicitud_Breve
    ])

    respuesta = str(respuesta.content).strip().capitalize()

    return {"respuesta" : respuesta}

def Analista(estado: Estado):
    usuario = estado["prompt_usuario"]
    instruccion_Analista = SystemMessage(content=system_Analista)
    solicitud_Analista = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_Analista,
        solicitud_Analista
    ])

    respuesta = str(respuesta.content).strip().capitalize()

    return {"respuesta_A" : respuesta}

def Plan(estado: Estado):
    usuario = estado["prompt_usuario"]
    analisis = estado["respuesta_A"]
    instruccion_Plan = SystemMessage(content=system_Plan)
    solicitud_Plan = HumanMessage(content=usuario+analisis)

    respuesta = llm.invoke([
        instruccion_Plan,
        solicitud_Plan
    ])

    respuesta = str(respuesta.content).strip().capitalize()

    return {"respuesta_P" : respuesta}

def Final(estado: Estado):
    usuario = estado["prompt_usuario"]
    analisis = estado["respuesta_A"]
    plan = estado["respuesta_P"]
    instruccion_Final = SystemMessage(content=system_Final)
    solicitud_Final = HumanMessage(content=usuario+analisis+plan)

    respuesta = llm.invoke([
        instruccion_Final,
        solicitud_Final
    ])

    respuesta = str(respuesta.content).strip().capitalize()

    return {"respuesta" : respuesta}

def seleccionar_agente(estado: Estado):
    return estado["especializado"]

grafo = StateGraph(Estado)
grafo.add_node("orquestador", orquestador)
grafo.add_node("Breve", Breve)
grafo.add_node("Analista", Analista)
grafo.add_node("Plan", Plan)
grafo.add_node("Final", Final)

grafo.add_edge(START, "orquestador")
grafo.add_conditional_edges("orquestador",seleccionar_agente,
                            {"Breve": "Breve", "Analista": "Analista"})
grafo.add_edge("Analista", "Plan")
grafo.add_edge("Plan", "Final")
grafo.add_edge("Final", END)
grafo.add_edge("Breve", END)

agente = grafo.compile()

imagen = agente.get_graph().draw_mermaid_png()
Image.open(BytesIO(imagen)).show()

while True:
    usuario = input("Usuario: ").strip()
    if usuario.lower() == "salir":
        break
    if not usuario:
        continue
    resultado = agente.invoke({"prompt_usuario": usuario})
    print("Agente seleccionado:", resultado["especializado"])
    print("Respuesta:", resultado["respuesta"])

 