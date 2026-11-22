import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

sys_prompt = SystemMessage(content="Analiza la intención de la consulta del usuario y clasifícala en una de estas dos categorías: 'Documental' (búsqueda de información o datos específicos) o 'Analista' (interpretación, comparación o razonamiento complejo). Responde única y exclusivamente con la palabra 'Documental' o 'Analista', sin explicaciones adicionales.")

solicitudUsuario = input("User: ").strip()

def orquestador(solicitudUsuario):
    instruccionOrquestador = SystemMessage(content=sys_prompt)
    solicitudOrquestador = HumanMessage(content=solicitudUsuario)
    respuesta = llm.invoke(instruccionOrquestador, solicitudOrquestador)

    if respuesta not in["Documental", "Analista"]:
        seleccion = "Analista"
    else:
        seleccion = respuesta.content
    return seleccion

def analista(solicitudUsuario):
    instruccionOrquestador = SystemMessage(content=sys_prompt)
    solicitudOrquestador = HumanMessage(content=solicitudUsuario)
    respuesta = llm.invoke(instruccionOrquestador, solicitudOrquestador)

    if respuesta not in["Documental", "Analista"]:
        seleccion = "Analista"
    else:
        seleccion = respuesta.content
    return seleccion

def documental(solicitudUsuario):
    instruccionOrquestador = SystemMessage(content=sys_prompt)
    solicitudOrquestador = HumanMessage(content=solicitudUsuario)
    respuesta = llm.invoke(instruccionOrquestador, solicitudOrquestador)

    if respuesta not in["Documental", "Analista"]:
        seleccion = "Analista"
    else:
        seleccion = respuesta.content
    return seleccion