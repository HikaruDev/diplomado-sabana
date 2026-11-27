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

instruccion = SystemMessage(content="Responde de forma clara y breve.")

Historial = []
Resumen = ""

while True:
    solicitud = input("\nUser: ").strip()

    if not solicitud:
        continue
    if solicitud.lower() == "salir":
        break

    pregunta = HumanMessage(content=f"{solicitud}\nResumen anterior: {Resumen}")
    respuesta = llm.invoke([instruccion, pregunta])

    # 1. Imprimir la respuesta inmediatamente al recibirla
    print(f"\nAI: {respuesta.content}\n")

    # 2. Actualizar el historial y el resumen en segundo plano
    Historial.extend([pregunta, respuesta])
    Resumen_1 = llm.invoke([
        SystemMessage(content=f"Resume la conversación en máximo 500 caracteres. Conserva la información importante. Resumen anterior: {Resumen}"),
        pregunta,
        respuesta,
        HumanMessage(content="Actualiza el resumen de la conversación.")
    ])
    Resumen = Resumen_1.content