import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

# Roles
rol_analista = SystemMessage(content="Evalúa con precisión la solicitud del usuario para verificar que sea clara, coherente y tenga sentido lógico. Identifica cualquier ambigüedad o contradicción antes de continuar.")
rol_planeador = SystemMessage(content="Diseña un plan de acción estructurado, lógico y paso a paso para resolver la solicitud del usuario. Define claramente las etapas necesarias antes de generar una respuesta.")
sys_rta_final = SystemMessage(content="Basándote estrictamente en el análisis previo y el plan definido, redacta la respuesta final para el usuario. Mantén un tono profesional, directo y coherente. Omite explicaciones sobre tu proceso interno y limita la extensión a un máximo de 100 palabras.")

Resumen = "Sin conversación previa."

while True:
    try:
        solicitud = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not solicitud or solicitud.lower() == "salir":
        break

    pregunta = HumanMessage(content=f"Solicitud: {solicitud}\nContexto previo: {Resumen}")

    # Análisis
    analisis = llm.invoke([rol_analista, pregunta])
    print(f"\nRespuesta rol analista: {analisis.content}")

    #Planeación
    plan = llm.invoke([
        rol_planeador,
        pregunta,
        HumanMessage(content=f"Análisis de la solicitud:\n{analisis.content}")
    ])
    print(f"\nRespuesta rol planeador: {plan.content}")

    #Respuesta Final
    respuesta_final = llm.invoke([
        sys_rta_final,
        pregunta,
        HumanMessage(content=f"Análisis:\n{analisis.content}\n\nPlan:\n{plan.content}")
    ])

    # 4. Actualizar resumen para la siguiente iteracion
    resumen_msg = llm.invoke([
        SystemMessage(content=f"Resume la conversación en máximo 500 caracteres conservando datos clave. Resumen anterior: {Resumen}"),
        pregunta,
        respuesta_final,
        HumanMessage(content="Actualiza el resumen.")
    ])
    Resumen = resumen_msg.content
    print(f"\nRespuesta final: {respuesta_final.content}")
