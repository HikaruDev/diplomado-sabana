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
    solicitud = input("User: ").strip()

    if solicitud.lower() == "salir":
        break

    pregunta = HumanMessage(content=solicitud + "\n Resumen Anterior:" + Resumen)

    respuesta = llm.invoke([instruccion] + [pregunta])

    Historial.extend([pregunta, respuesta])

    Resumen_aux = [pregunta, respuesta]

    Resumen_1 = llm.invoke(
        [SystemMessage(content="Resume la conversación en máximo 500 caracteres. Conserva la información importante. Resumen anterior: " + Resumen)]
        + Resumen_aux
        + [HumanMessage(content="Actualiza el resumen de la conversación.")]
    )

    Resumen = Resumen_1.content

    print(respuesta.content)