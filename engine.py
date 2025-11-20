from openai import OpenAI
import os
from ingest.local_storage_vector import search_documents
import json
from guardrails import guarded_response

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_context(docs, link_flag):
    context = ""

    for i, doc in enumerate(docs, 1):
        context += f"RECEPT {i}:\n"
        context += f"Naslov: {doc.get('title')}\n"

        if link_flag:
            source = doc.get("source")
            if source:
                context += f"[Otvori recept]({source})\n"

        context += f"Sadržaj:\n{doc.get('content')}\n\n"

    return context

def pronadji_recepte(upit: str, link_flag:bool):
    docs = search_documents(upit, k=3)

    # vraćamo samo kontekst, ne cijeli object
    context = build_context(docs, link_flag)
    

    return {
        "kontekst": context
    }


def generate_answer(question: str) -> str:
    #FUNCTION CALLING
    tools = [
        {
            "type": "function",
            "function": {
                "name": "pronadji_recepte",
                "description": "Pretražuje bazu recepata i vraća relevantan sadržaj",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "upit": {
                            "type": "string",
                            "description": "Upit korisnika za pretragu recepata"
                        },
                            "prikazi_linkove": {
                            "type": "boolean",
                            "description": "Ako je true, prikaži klikabilne linkove do recepta"
                        }
                    },
                    "required": ["upit"]
                }
            }
        }
    ]

    system_prompt = """
                    Ti si CookBot, stručni kulinarski asistent.
                    Koristi samo informacije iz konteksta recepata.
                    Odgovaraj jasno, praktično i razumljivo.
                    Ako korisnik traži link, izvor, gdje može pogledati recept ili želi da ga otvori,
                    postavi prikazi_linkove = true.
                    Ako ne traži link eksplicitno, prikaži samo tekst recepta.
                    """

    # Prvi poziv  modela - model odlučuje da li treba FAISS
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        tools=tools,
        tool_choice="auto"
    )

    

    msg = response.choices[0].message

    # 2. Ako model traži funkciju → zovi FAISS
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        rezultat = pronadji_recepte(args["upit"], args.get("prikazi_linkove", False))
        # 3. Drugi poziv: daj mu kontekst i generiši finalni odgovor
        final_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Koristi samo dostavljeni kontekst za odgovor."},
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "tool_calls": msg.tool_calls
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(rezultat, ensure_ascii=False)
                }
            ]
        )

        raw_answer = final_response.choices[0].message.content
        safe_answer = guarded_response(question, raw_answer)
        return safe_answer
    
    raw_answer = msg.content
    safe_answer = guarded_response(question, raw_answer)
    return safe_answer

    
    

