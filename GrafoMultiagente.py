import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from PIL import Image

load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

#Metaprompts
sys_prompt_orq = SystemMessage(
    content=(
        "Eres el nodo Orquestador de un sistema multiagente. Analiza la solicitud del usuario y el historial actual.\n"
        "Tu tarea es decidir el siguiente paso:\n"
        "- 'A': Si es una consulta general, definición o explicación conceptual.\n"
        "- 'B': Si requiere análisis de un problema, requerimientos o diseño de una solución.\n"
        "- 'C': Si el usuario pide explícitamente evaluar o auditar un texto/solución ya entregada en su prompt.\n"
        "- 'END': Si el agente 'A' (Chat) ya respondió satisfactoriamente a la consulta general del usuario.\n\n"
        "Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta (sin markdown ni texto extra):\n"
        '{"respuesta": "Breve justificación de tu decisión", "siguiente_ruta": "A" | "B" | "C" | "END"}'
    )
)

sys_prompt_agente_a = SystemMessage(
    content=(
        "Eres el Agente A (Chat). Tu rol es responder consultas generales, definiciones y explicaciones "
        "de conceptos de manera clara, precisa y directa.\n"
        "Siempre debes devolver el control al 'orquestador' tras responder.\n\n"
        "Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta (sin markdown ni texto extra):\n"
        '{"respuesta": "Tu explicación o respuesta completa aquí", "siguiente_ruta": "orquestador"}'
    )
)

sys_prompt_agente_b = SystemMessage(
    content=(
        "Eres el Agente B (Analista). Tu rol es analizar problemas, identificar requerimientos y desarrollar "
        "soluciones o propuestas estructuradas.\n"
        "- Si existe retroalimentación previa del Evaluador (Agente C), debes corregir y mejorar tu propuesta.\n"
        "- Decide 'siguiente_ruta':\n"
        "   * Usa 'C' si el usuario pidió evaluación/revisión/auditoría, o si el problema es crítico y requiere validación.\n"
        "   * Usa 'END' si el usuario solo pidió un análisis o propuesta directa sin solicitar evaluación.\n\n"
        "Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta (sin markdown ni texto extra):\n"
        '{"respuesta": "Tu análisis y solución estructurada aquí", "siguiente_ruta": "C" | "END"}'
    )
)

sys_prompt_agente_c = SystemMessage(
    content=(
        "Eres el Agente C (Evaluador). Tu rol es revisar soluciones propuestas, identificar errores, "
        "limitaciones y oportunidades de mejora.\n"
        "- Decide 'siguiente_ruta':\n"
        "   * Usa 'B' si encuentras fallos graves, omisiones importantes o falta de claridad que el Analista deba corregir.\n"
        "   * Usa 'END' si la propuesta es sólida, completa y cumple con lo solicitado (incluye tus observaciones finales en 'respuesta').\n\n"
        "Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta (sin markdown ni texto extra):\n"
        '{"respuesta": "Tu evaluación detallada o lista de correcciones requeridas", "siguiente_ruta": "B" | "END"}'
    )
)

#Estados
class Estado(TypedDict, total=False):
    prompt_usuario: str
    ruta_actual: str
    respuesta_chat: str
    analisis_solucion: str
    evaluacion: str
    respuesta_final: str
    iteraciones_revision: int

#Json
def extraer_json(texto: str, ruta_fallback: str) -> dict:
    """Extrae y parsea el JSON generado por el LLM de forma segura."""
    limpio = re.sub(r"```json\s*|\s*```", "", texto).strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", limpio, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"respuesta": limpio, "siguiente_ruta": ruta_fallback}


#Nodos
def nodo_orquestador(estado: Estado):
    contexto = f"Solicitud del usuario: {estado['prompt_usuario']}"
    if estado.get("respuesta_chat"):
        contexto += f"\n\nRespuesta ya generada por Agente A (Chat):\n{estado['respuesta_chat']}"

    respuesta = llm.invoke([sys_prompt_orq, HumanMessage(content=contexto)])
    datos = extraer_json(respuesta.content, "END")

    ruta = datos.get("siguiente_ruta", "END")
    if ruta not in ["A", "B", "C", "END"]:
        ruta = "END"

    print(f"\n[ORQUESTADOR] -> Ruta elegida: {ruta}")
    print(f"  Motivo: {datos.get('respuesta')}")

    actualizacion = {"ruta_actual": ruta}
    if ruta == "END" and estado.get("respuesta_chat"):
        actualizacion["respuesta_final"] = estado["respuesta_chat"]
    return actualizacion


def nodo_agente_a(estado: Estado):
    respuesta = llm.invoke([sys_prompt_agente_a, HumanMessage(content=estado["prompt_usuario"])])
    datos = extraer_json(respuesta.content, "orquestador")

    contenido = datos.get("respuesta", "")
    print(f"\n[AGENTE A - CHAT] -> Ruta elegida: orquestador")
    print(f"  Respuesta:\n{contenido}")

    return {
        "respuesta_chat": contenido,
        "ruta_actual": "orquestador",
    }


def nodo_agente_b(estado: Estado):
    contexto = f"Solicitud del usuario: {estado['prompt_usuario']}"
    if estado.get("evaluacion"):
        contexto += (
            f"\n\nTu propuesta anterior fue:\n{estado.get('analisis_solucion')}"
            f"\n\nRetroalimentación del Evaluador (Agente C) que debes corregir:\n{estado['evaluacion']}"
        )

    respuesta = llm.invoke([sys_prompt_agente_b, HumanMessage(content=contexto)])
    datos = extraer_json(respuesta.content, "END")

    ruta = datos.get("siguiente_ruta", "END")
    if ruta not in ["C", "END"]:
        ruta = "END"

    contenido = datos.get("respuesta", "")
    print(f"\n[AGENTE B - ANALISTA] -> Ruta elegida: {ruta}")
    print(f"  Propuesta:\n{contenido}")

    return {
        "analisis_solucion": contenido,
        "respuesta_final": contenido,
        "ruta_actual": ruta,
    }


def nodo_agente_c(estado: Estado):
    iteraciones = estado.get("iteraciones_revision", 0) + 1
    contexto = (
        f"Solicitud original del usuario: {estado['prompt_usuario']}\n\n"
        f"Propuesta del Analista (Agente B) a evaluar:\n{estado.get('analisis_solucion', 'No provista')}"
    )

    respuesta = llm.invoke([sys_prompt_agente_c, HumanMessage(content=contexto)])
    datos = extraer_json(respuesta.content, "END")

    ruta = datos.get("siguiente_ruta", "END")
    
    # Límite de seguridad de 2 ciclos de mejora para evitar bucles infinitos
    if ruta not in ["B", "END"] or iteraciones >= 2:
        ruta = "END"

    contenido = datos.get("respuesta", "")
    print(f"\n[AGENTE C - EVALUADOR] (Ciclo {iteraciones}) -> Ruta elegida: {ruta}")
    print(f"  Evaluación:\n{contenido}")

    resultado_final = (
        f"### Propuesta Final (Analista):\n{estado.get('analisis_solucion', '')}\n\n"
        f"### Veredicto de Calidad (Evaluador):\n{contenido}"
    )

    return {
        "evaluacion": contenido,
        "respuesta_final": resultado_final,
        "ruta_actual": ruta,
        "iteraciones_revision": iteraciones,
    }



#Contrucción y compilacioon del grafo
def decidir_ruta(estado: Estado) -> str:
    return estado["ruta_actual"]

builder = StateGraph(Estado)

builder.add_node("orquestador", nodo_orquestador)
builder.add_node("A", nodo_agente_a)
builder.add_node("B", nodo_agente_b)
builder.add_node("C", nodo_agente_c)

# Inicio
builder.add_edge(START, "orquestador")

# Salidas condicionales del Orquestador
builder.add_conditional_edges(
    "orquestador",
    decidir_ruta,
    {
        "A": "A",
        "B": "B",
        "C": "C",
        "END": END,
    },
)

# Agente A siempre vuelve al Orquestador (según el diagrama)
builder.add_edge("A", "orquestador")

# Salidas condicionales del Agente B (Revisión -> C | END)
builder.add_conditional_edges(
    "B",
    decidir_ruta,
    {
        "C": "C",
        "END": END,
    },
)

# Salidas condicionales del Agente C (Mejora -> B | END)
builder.add_conditional_edges(
    "C",
    decidir_ruta,
    {
        "B": "B",
        "END": END,
    },
)

agente = builder.compile()

#Salida
try:
    #Creacion del grafo en grafico
    imagen_bytes = agente.get_graph().draw_mermaid_png()
    ruta_img = Path(__file__).resolve().parent / "grafo_multiagente.png"
    with open(ruta_img, "wb") as f:
        f.write(imagen_bytes)
    print(f"[INFO] Diagrama del grafo guardado en: {ruta_img}")
except Exception as e:
    print(f"[WARN] No se pudo generar la imagen PNG ({e}). Estructura Mermaid:")
    print(agente.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("\n=== SISTEMA MULTIAGENTE INICIADO (Escribe 'salir' para terminar) ===")
    while True:
        usuario = input("\nUser: ").strip()
        if usuario.lower() == "salir":
            break
        if not usuario:
            continue

        resultado = agente.invoke({"prompt_usuario": usuario, "iteraciones_revision": 0})
        print("\n==================================================")
        print("RESPUESTA FINAL ENTREGADA AL USUARIO:")
        print("==================================================")
        print(resultado.get("respuesta_final", "Sin respuesta generada."))
        print("==================================================\n")