"""
chunker.py

Splits loaded markdown documents into text chunks for embedding and vector search.
"""

from typing import List, Dict, Any


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split markdown documents into smaller text chunks based on section headers and paragraph bounds.

    Returns
    -------
    List[Dict[str, Any]]
        List of chunk objects containing 'chunk_id', 'source', 'category', and 'text'.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        content = doc["content"]
        source = doc["source"]
        category = doc["category"]

        # Split by section headers (## or ###)
        sections = content.split("\n## ")
        for sec_idx, sec in enumerate(sections):
            sec_text = ("## " + sec) if sec_idx > 0 else sec
            sec_text = sec_text.strip()
            if not sec_text:
                continue

            # If section is small, keep as single chunk
            if len(sec_text) <= chunk_size:
                chunks.append({
                    "chunk_id": f"{doc['id']}#sec_{sec_idx}",
                    "source": source,
                    "category": category,
                    "text": sec_text
                })
            else:
                # Sub-split long section by paragraphs
                paragraphs = sec_text.split("\n\n")
                current_chunk = ""
                sub_idx = 0

                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        current_chunk = (current_chunk + "\n\n" + para).strip()
                    else:
                        if current_chunk:
                            chunks.append({
                                "chunk_id": f"{doc['id']}#sec_{sec_idx}_p{sub_idx}",
                                "source": source,
                                "category": category,
                                "text": current_chunk
                            })
                            sub_idx += 1
                        current_chunk = para.strip()

                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{doc['id']}#sec_{sec_idx}_p{sub_idx}",
                        "source": source,
                        "category": category,
                        "text": current_chunk
                    })

    print(f"[RAG Chunker] Created {len(chunks)} text chunks from {len(documents)} documents.")
    return chunks
