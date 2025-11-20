"""
Guardrails modul za CookBot.
Koristi LLM da evaluira odgovor modela i odluči da li je siguran i relevantan.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def guard_answer(question: str, answer: str) -> dict:
    """
    Evaluira odgovor modela i vraća odluku da li je siguran za prikaz.

    Returns:
        {
          "allowed": bool,
          "reason": str
        }
    """

    system_prompt = """
Ti si strogi evaluator sigurnosti za AI kulinarskog asistenta.

Provjeri odgovor prema pravilima:
- Odgovor mora biti vezan isključivo za kuhanje i recepte
- Ne smije sadržavati medicinske savjete
- Ne smije sadržavati opasne postupke
- Ne smije izmišljati recepte
- Ne smije biti uvredljiv, toksičan ili neprimjeren

Ako je odgovor siguran napiši:
SAFE

Ako nije, napiši:
UNSAFE: <kratko objašnjenje>
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"PITANJE: {question}\nODGOVOR: {answer}"}
        ],
        temperature=0
    )

    evaluation = response.choices[0].message.content.strip()

    if evaluation.startswith("SAFE"):
        return {
            "allowed": True,
            "reason": ""
        }

    return {
        "allowed": False,
        "reason": evaluation.replace("UNSAFE:", "").strip()
    }


# Pomoćni fallback odgovor

def guarded_response(question: str, answer: str) -> str:
    result = guard_answer(question, answer)

    if result["allowed"]:
        return answer

    return (
        "Izvinjavam se, ali odgovor ne zadovoljava sigurnosne kriterije. "
        "Molimo pokušajte sa drugačijim pitanjem.\n"
        f"Razlog: {result['reason']}"
    )
