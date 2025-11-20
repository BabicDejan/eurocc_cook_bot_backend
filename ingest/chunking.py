"""
Pametni chunking prilagođen za recepte (Cook-bot).
Zadržava logiku sastojaka i koraka pripreme.
"""
import re
from typing import List


def chunk(text: str, max_len: int = 1200, overlap: int = 150) -> List[str]:
    """
    Chunking optimizovan za kulinarske recepte.

    - Prvo dijeli po sekcijama (Sastojci / Priprema)
    - Onda dijeli po rečenicama uz overlap
    """
    
    sections = split_sections(text)
    final_chunks = []

    for section_title, section_text in sections:
        chunks = chunk_by_sentences(section_text, max_len, overlap)

        for c in chunks:
            final_chunks.append(f"{section_title}\n{c}")

    return final_chunks


def split_sections(text: str):
    """
    Dijeli tekst po logičkim dijelovima recepta.
    """
    parts = []
    patterns = ["Sastojci:", "Priprema:"]

    current_title = "Opis recepta"
    buffer = []

    for line in text.splitlines():
        if any(p in line for p in patterns):
            if buffer:
                parts.append((current_title, "\n".join(buffer)))
                buffer = []
            current_title = line.strip()
        else:
            buffer.append(line)

    if buffer:
        parts.append((current_title, "\n".join(buffer)))

    return parts


def chunk_by_sentences(text: str, max_len: int, overlap: int):
    """
    Klasični sentence-based chunking sa overlapom.
    """
    sentences = re.split(r'(?<=[\\.!?])\\s+', text.strip())
    
    buf = []
    cur = 0
    chunks = []

    for s in sentences:
        if cur + len(s) > max_len and buf:
            chunk_text = " ".join(buf)
            chunks.append(chunk_text)

            tail = chunk_text[-overlap:]
            buf = [tail, s]
            cur = len(tail) + len(s)
        else:
            buf.append(s)
            cur += len(s)

    if buf:
        chunks.append(" ".join(buf))

    return chunks
