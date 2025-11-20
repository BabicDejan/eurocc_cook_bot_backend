"""
Generisanje vektorskih embeddinga za sve dokumente koriscenjem OpenAI API-ja + FAISS biblioteka za brzu pretragu .
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from local_storage_vector import build_vector_index

if __name__ == "__main__":
    print("="*70)
    print("BUILDING VECTOR INDEX FOR SEMANTIC SEARCH")
    print("="*70)
    print()
    print("This will:")
    print("  1. Load all documents from data/parsed_data.json")
    print("  2. Generate OpenAI embeddings for each document")
    print("  3. Build FAISS index for fast semantic search")
    print()
    
    build_vector_index()
    
    print()
    print("="*70)
    print("✓ Vector index built successfully!")
    print("="*70)
    print()
    print("You can now use semantic search in your chatbot!")
    print()

