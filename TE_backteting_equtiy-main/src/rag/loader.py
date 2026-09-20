"""
loader.py

Loads markdown files from all knowledge_base directories and repository documentation
with rich metadata extraction (source, category, topic, section, version).
"""

import os
from typing import List, Dict, Any


def load_knowledge_base_docs(kb_dir: str = None) -> List[Dict[str, Any]]:
    """
    Recursively load all markdown documents across knowledge_base subdirectories and docs.

    Returns
    -------
    List[Dict[str, Any]]
        List of document objects containing 'id', 'source', 'category', 'topic', 'version', and 'content'.
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
                    rel_path = os.path.relpath(full_path, base_dir).replace("\\", "/")
                    
                    # Extract category & topic from directory structure
                    parent_dir = os.path.basename(os.path.dirname(full_path))
                    topic_name = os.path.splitext(file)[0]

                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        documents.append({
                            "id": file,
                            "source": rel_path,
                            "category": parent_dir,
                            "topic": topic_name,
                            "version": "1.0",
                            "content": content
                        })
                    except Exception as e:
                        print(f"[RAG Loader Warning] Could not read {full_path}: {e}")

    # Add root README / system report files if available
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
                    "topic": os.path.splitext(standalone)[0],
                    "version": "1.0",
                    "content": content
                })
            except Exception:
                pass

    print(f"[RAG Loader] Successfully loaded {len(documents)} markdown knowledge base documents.")
    return documents
