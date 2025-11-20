"""
Vector-based storage za parsed data koristeći FAISS + OpenAI embeddings.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import hashlib

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Storage files
STORAGE_FILE = DATA_DIR / "cookbot_data.json"
VECTOR_INDEX_FILE = DATA_DIR / "vector_index.faiss"
DOCS_METADATA_FILE = DATA_DIR / "docs_metadata.pkl"
EMBEDDING_CACHE_FILE = DATA_DIR / "embedding_cache.pkl"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# Global embedding cache (in-memory)
_embedding_cache = {}


def load_embedding_cache():
    """Učitaj embedding cache ako postoji."""
    global _embedding_cache
    if EMBEDDING_CACHE_FILE.exists():
        try:
            with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                _embedding_cache = pickle.load(f)
                print(f"Loaded {len(_embedding_cache)} cached embeddings")
        except:
            _embedding_cache = {}


def save_embedding_cache():
    """Sačuvaj embedding cache."""
    global _embedding_cache
    if _embedding_cache:
        EMBEDDING_CACHE_FILE.parent.mkdir(exist_ok=True)
        with open(EMBEDDING_CACHE_FILE, 'wb') as f:
            pickle.dump(_embedding_cache, f)


# Load cache on import
load_embedding_cache()

def get_embedding(text: str) -> np.ndarray:
    """Generiše embedding za tekst koristeći OpenAI sa CACHING."""
    global _embedding_cache
    
    # Check cache first
    cache_key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    # Generate embedding
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Cache it
        _embedding_cache[cache_key] = embedding
        
        # Save cache every 10 new entries
        if len(_embedding_cache) % 10 == 0:
            save_embedding_cache()
        
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def load_documents() -> List[Dict]:
    """Učitaj dokumente iz JSON."""
    if not STORAGE_FILE.exists():
        return []
    
    with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_documents(docs: List[Dict]):
    """Sačuvaj dokumente u JSON."""
    STORAGE_FILE.parent.mkdir(exist_ok=True)
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} documents to {STORAGE_FILE}")


def build_vector_index():
    """
    Kreira ili regeneriše FAISS index sa embeddings.
    Ovoga puta će koristiti samo title + content za embedding.
    """
    print("Building vector index...")
    
    docs = load_documents()
    if not docs:
        print("No documents to index!")
        return
    
    print(f"Generating embeddings for {len(docs)} documents...")
    
    # Kreiraj FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product za cosine similarity
    
    embeddings = []
    metadata = []
    
    for i, doc in enumerate(docs):
        # Kombinuj title + content za embedding
        text_to_embed = f"{doc.get('title','')} {doc.get('category','')} {doc.get('content','')}"
        
        # Limit na 8000 karaktera zbog embedding API limita
        text_to_embed = text_to_embed[:8000]
        
        embedding = get_embedding(text_to_embed)
        embeddings.append(embedding)
        metadata.append({
            'doc_id': i,
            'title': doc.get('title', ''),
            'url': doc.get('source', ''),
            'type': doc.get('type', 'recipe'),
            'category': doc.get('category', '')
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(docs)} documents...")
    
    # Normalizuj embeddings za cosine similarity
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    # Save index and metadata
    VECTOR_INDEX_FILE.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(VECTOR_INDEX_FILE))
    with open(DOCS_METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"SUCCESS: Vector index saved with {len(docs)} documents indexed")


def search_documents(query: str, k: int = 8):
    if not VECTOR_INDEX_FILE.exists() or not DOCS_METADATA_FILE.exists():
        build_vector_index()

    index = faiss.read_index(str(VECTOR_INDEX_FILE))

    with open(DOCS_METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)

    docs = load_documents()

    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, min(k * 2, len(metadata)))

    results = []
    seen_sources = set()

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        doc_id = meta.get("doc_id")

        if doc_id is None or doc_id >= len(docs):
            continue

        doc = docs[doc_id]

        if not isinstance(doc, dict):
            continue

        source = doc.get("source", "")

        if source and source in seen_sources:
            continue

        seen_sources.add(source)
        results.append(doc)

        if len(results) >= k:
            break

    return results


def hash_content(content: str) -> str:
    """Hash content za uklanjanje duplikacije"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


if __name__ == "__main__":
    # Build index kada se pokrene direktno
    build_vector_index()

