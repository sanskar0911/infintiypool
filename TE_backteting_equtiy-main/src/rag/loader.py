"""
loader.py

Loads markdown files from the knowledge_base directory.
"""

import os
from typing import List, Dict, Any


def load_knowledge_base_docs(kb_dir: str = None) -> List[Dict[str, Any]]:
    """
    Recursively load all markdown files from knowledge_base directory, docs directory,
    and root documentation files into the RAG vector store.

    Returns
    -------
    List[Dict[str, Any]]
        List of document objects with 'id', 'source', 'category', and 'content'.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    search_dirs = [
        os.path.join(base_dir, "knowledge_base"),
        os.path.join(base_dir, "docs")
    ]
    if kb_dir and kb_dir not in search_dirs:
        search_dirs.append(kb_dir)

    documents: List[Dict[str, Any]] = []

    for target_dir in search_dirs:
        if not os.path.exists(target_dir):
            continue
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    category = os.path.basename(os.path.dirname(full_path)) or "general"
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        documents.append({
                            "id": file,
                            "source": rel_path.replace("\\", "/"),
                            "category": category,
                            "content": content
                        })
                    except Exception as e:
                        print(f"[RAG Loader Warning] Could not read {full_path}: {e}")

    # Add standalone root documentation files if present
    for standalone in ["README.md", "AI_CODING_REPORT.md"]:
        s_path = os.path.join(base_dir, standalone)
        if os.path.exists(s_path):
            try:
                with open(s_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                documents.append({
                    "id": standalone,
                    "source": standalone,
                    "category": "project_overview",
                    "content": content
                })
            except Exception:
                pass

    print(f"[RAG Loader] Successfully loaded {len(documents)} knowledge base documents across repository.")
    return documents
