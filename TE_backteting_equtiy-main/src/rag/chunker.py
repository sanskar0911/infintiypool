"""
chunker.py

Splits loaded markdown documents into text chunks for embedding and vector search,
preserving metadata (source, category, topic, section, version) and contextual headers.
"""

from typing import List, Dict, Any


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Split markdown documents into smaller text chunks based on section headers and paragraph bounds.
    Ensures every chunk contains full narrative context.

    Returns
    -------
    List[Dict[str, Any]]
        List of chunk objects containing 'chunk_id', 'source', 'category', 'topic', 'section', 'version', and 'text'.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        content = doc.get("content", "")
        source = doc.get("source", "unknown")
        category = doc.get("category", "general")
        topic = doc.get("topic", "general")
        version = doc.get("version", "1.0")

        if not content.strip():
            continue

        # Extract main document title from first # line if present
        doc_lines = [l.strip() for l in content.split("\n") if l.strip()]
        doc_title = topic.replace("_", " ").title()
        if doc_lines and doc_lines[0].startswith("# "):
            doc_title = doc_lines[0].replace("# ", "").strip()

        # Split by markdown section headers (## or ###)
        sections = content.split("\n## ")
        for sec_idx, sec in enumerate(sections):
            sec_text = ("## " + sec) if sec_idx > 0 else sec
            sec_text = sec_text.strip()
            if not sec_text:
                continue

            # Strip standalone title-only section 0 if it has no narrative content
            lines = [l.strip() for l in sec_text.split("\n") if l.strip()]
            narrative_lines = [l for l in lines if not l.startswith("#") and not l.startswith("---") and len(l) > 10]
            if not narrative_lines:
                continue

            # Extract section title from first line
            first_line = sec_text.split("\n")[0].replace("#", "").strip()
            section_title = first_line if first_line else f"Section_{sec_idx}"

            # Prepend Document Title to chunk text if not present for contextual richness
            formatted_text = sec_text
            if not formatted_text.startswith(f"# {doc_title}"):
                formatted_text = f"# {doc_title}\n{sec_text}"

            # If section is within chunk_size, keep as single chunk
            if len(formatted_text) <= chunk_size:
                chunks.append({
                    "chunk_id": f"{doc['id']}#sec_{sec_idx}",
                    "source": source,
                    "category": category,
                    "topic": topic,
                    "section": section_title,
                    "version": version,
                    "text": formatted_text
                })
            else:
                # Sub-split long section by paragraphs
                paragraphs = sec_text.split("\n\n")
                current_chunk = f"# {doc_title}\n## {section_title}"
                sub_idx = 0

                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        current_chunk = (current_chunk + "\n\n" + para).strip()
                    else:
                        if len(current_chunk) > 40:
                            chunks.append({
                                "chunk_id": f"{doc['id']}#sec_{sec_idx}_p{sub_idx}",
                                "source": source,
                                "category": category,
                                "topic": topic,
                                "section": section_title,
                                "version": version,
                                "text": current_chunk
                            })
                            sub_idx += 1
                        current_chunk = f"# {doc_title}\n" + para.strip()

                if len(current_chunk) > 40:
                    chunks.append({
                        "chunk_id": f"{doc['id']}#sec_{sec_idx}_p{sub_idx}",
                        "source": source,
                        "category": category,
                        "topic": topic,
                        "section": section_title,
                        "version": version,
                        "text": current_chunk
                    })

    print(f"[RAG Chunker] Created {len(chunks)} text chunks from {len(documents)} documents (chunk_size={chunk_size}).")
    return chunks
