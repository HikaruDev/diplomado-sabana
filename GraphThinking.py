import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

#from PIL import Image
#from io import BytesIO


load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

sys_prompt_orq = SystemMessage(
    content=(
        "Evalúa la intención de la consulta del usuario. "
        "Si es simple o directa, responde únicamente: RtaBreve. "
        "Si requiere análisis y planificación, responde únicamente: Analizar."
    )
)

sys_prompt_rtaBreve = SystemMessage(
    content=(
        "Responde de forma directa, clara y precisa a la solicitud del usuario en un máximo de 200 palabras. "
        "Ve directo al grano, evita introducciones innecesarias y resuelve completamente la consulta."
    )
)

sys_prompt_analista = SystemMessage(
    content=(
        "Evalúa con precisión la solicitud del usuario para verificar que sea clara, coherente "
        "y tenga sentido lógico. Identifica cualquier ambigüedad o contradicción antes de continuar."
    )
)

sys_prompt_planeador = SystemMessage(
    content=(
        "Diseña un plan de acción estructurado, lógico y paso a paso para resolver la solicitud del usuario, "
        "tomando en cuenta el análisis previo."
    )
)

sys_prompt_rtaFinal = SystemMessage(
    content=(
        "Basándote estrictamente en el análisis previo y el plan definido, "
        "redacta la respuesta final para el usuario. Mantén un tono profesional, directo y coherente. "
        "Omite explicaciones sobre tu proceso interno y limita la extensión a un máximo de 100 palabras."
    )
)

class Estado(TypedDict, total=False):
    prompt_usuario: str
    categoria: str
    analisis: str
    plan: str
    respuesta_final: str

def orquestador(estado: Estado):
    respuesta = llm.invoke([sys_prompt_orq, HumanMessage(content=estado["prompt_usuario"])])
    seleccion = respuesta.content.strip()
    if seleccion not in ["RtaBreve", "Analizar"]:
        seleccion = "RtaBreve"
    return {"categoria": seleccion}

def nodo_rtaBreve(estado: Estado):
    respuesta = llm.invoke([sys_prompt_rtaBreve, HumanMessage(content=estado["prompt_usuario"])])
    return {"respuesta_final": respuesta.content.strip()}

def nodo_analisis(estado: Estado):
    respuesta = llm.invoke([sys_prompt_analista, HumanMessage(content=estado["prompt_usuario"])])
    return {"analisis": respuesta.content.strip()}

def nodo_planeacion(estado: Estado):
    contexto = f"Solicitud: {estado['prompt_usuario']}\n\nAnálisis previo:\n{estado['analisis']}"
    respuesta = llm.invoke([sys_prompt_planeador, HumanMessage(content=contexto)])
    return {"plan": respuesta.content.strip()}

def nodo_respuestaFinal(estado: Estado):
    contexto = (
        f"Solicitud: {estado['prompt_usuario']}\n\n"
        f"Análisis:\n{estado['analisis']}\n\n"
        f"Plan:\n{estado['plan']}"
    )
    respuesta = llm.invoke([sys_prompt_rtaFinal, HumanMessage(content=contexto)])
    return {"respuesta_final": respuesta.content.strip()}

def decidir_ruta(estado: Estado):
    return estado["categoria"]

#Construcción y compilación del grafo
builder = StateGraph(Estado)

builder.add_node("orquestador", orquestador)
builder.add_node("RtaBreve", nodo_rtaBreve)
builder.add_node("Analizar", nodo_analisis)
builder.add_node("Planear", nodo_planeacion)
builder.add_node("RtaFinal", nodo_respuestaFinal)

builder.add_edge(START, "orquestador")

builder.add_conditional_edges(
    "orquestador",
    decidir_ruta,
    {
        "RtaBreve": "RtaBreve",
        "Analizar": "Analizar",
    },
)

# Camino corto
builder.add_edge("RtaBreve", END)

# Camino largo
builder.add_edge("Analizar", "Planear")
builder.add_edge("Planear", "RtaFinal")
builder.add_edge("RtaFinal", END)

agente = builder.compile()

#imagen = agente.get_graph().draw_mermaid_png()
#Image.open(BytesIO(imagen)).show()

while True:
    usuario = input("User: ").strip()
    if usuario.lower() == "salir":
        break
    if not usuario:
        continue

    resultado = agente.invoke(
        {"prompt_usuario": usuario}
    )
    
    print(f"\n[Ruta elegida]: {resultado['categoria']}")
    
    if resultado["categoria"] == "Analizar":
        print(f"\n--- 1. SALIDA NODO ANALIZAR ---\n{resultado.get('analisis')}")
        print(f"\n--- 2. SALIDA NODO PLANEAR ---\n{resultado.get('plan')}")

    # Esta línea debe ir indentada dentro del while:
    print(f"\n--- 3. RESPUESTA FINAL ---\n{resultado['respuesta_final']}\n")