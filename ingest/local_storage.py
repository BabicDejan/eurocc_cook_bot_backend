"""
Local storage za Cook-bot parsirane podatke.
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict

STORAGE_FILE = Path("../data/cookbot_data.json")


def save_documents(new_docs: List[Dict]):
    """
    Dodaje nove dokumente u postojeću bazu uz preskakanje duplikata
    Ovo je korisno kada imate zakazano scrape-ovanje (npr svako nedjelju dana)
    """
    existing_docs = load_documents()
    existing_hashes = {hash_content(d.get("content", "")) for d in existing_docs}

    added = 0
    for doc in new_docs:
        h = hash_content(doc.get("content", ""))
        if h not in existing_hashes:
            doc["content_hash"] = h
            existing_docs.append(doc)
            added += 1

    STORAGE_FILE.parent.mkdir(exist_ok=True)
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_docs, f, ensure_ascii=False, indent=2)

    print(f"Added {added} new documents. Total: {len(existing_docs)}")


def load_documents() -> List[Dict]:
    if not STORAGE_FILE.exists():
        return []

    with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def search_documents(query: str, k: int = 8) -> List[Dict]:
    """
    Pretraga optimizovana za recepte (bez news logike).
    """
    docs = load_documents()
    if not docs:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_docs = []

    for doc in docs:
        content_lower = doc.get("content", "").lower()
        title_lower = doc.get("title", "").lower()

        score = 0
        content_words = set(content_lower.split())
        title_words = set(title_lower.split())

        score += len(query_words & title_words) * 3
        score += len(query_words & content_words)

        if query_lower in content_lower or query_lower in title_lower:
            score += 5

        # Bonus za istu kategoriju ako postoji u upitu
        category = doc.get("category", "").lower()
        if category and category in query_lower:
            score += 2

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs if score > 0][:k]


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
