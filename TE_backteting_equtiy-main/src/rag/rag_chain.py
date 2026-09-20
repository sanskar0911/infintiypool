"""
rag_chain.py

Grounded RAG Q&A chain combining Multilingual MADRID translation, vector search,
LLM synthesis, hallucination control, token accounting, and pipeline stage observability.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from rag.retriever import get_retriever, RAGRetriever
from rag.translator.translator import MultilingualMADRIDTranslator
from rag.translator.schemas import RAGPipelineResult


def clean_chunks_to_paragraphs(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Extracts substantive narrative paragraphs from retrieved chunks,
    prioritizing definitions and core concepts, and filtering out header tags or list noise.
    """
    definition_paras = []
    other_paras = []

    for c in retrieved_chunks:
        section = c.get("section", "").lower()
        text = c.get("text", "")
        lines = text.split("\n")
        clean_lines = []

        for line in lines:
            line_str = line.strip()
            # Skip headers, dividers, bullet lists, and code blocks
            if not line_str or line_str.startswith("#") or line_str.startswith("---") or line_str.startswith("`"):
                continue
            if line_str.startswith("- ") and len(line_str) < 35:
                continue
            clean_lines.append(line_str)

        if clean_lines:
            para = " ".join(clean_lines)
            if len(para) > 40:
                if "definition" in section or "core concept" in section:
                    if para not in definition_paras:
                        definition_paras.append(para)
                else:
                    if para not in other_paras and para not in definition_paras:
                        other_paras.append(para)

    all_paras = definition_paras + other_paras
    if not all_paras:
        all_paras = [c.get("text", "").strip() for c in retrieved_chunks if c.get("text")]

    return "\n\n".join(all_paras)


class RAGChain:
    """Complete 8-Stage Multilingual RAG + MADRID Pipeline Coordinator."""

    def __init__(self, retriever: RAGRetriever = None):
        self.retriever = retriever or get_retriever()
        self.translator = MultilingualMADRIDTranslator()
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def execute_pipeline(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Executes complete 8-stage pipeline:
        1. Ingest User Input
        2. Detect Language
        3. Convert Input -> MADRID
        4. Vector RAG Retrieval
        5. Filter Context Chunks
        6. LLM Context Synthesis (Output in MADRID + Target Language)
        7. Decode MADRID -> Target Language in proper paragraphs
        8. Return Payload + Token Accounting + Latencies

        Returns
        -------
        Dict[str, Any]
            Full pipeline state dictionary conforming to UI dashboard expectations.
        """
        start_total = time.time()
        latencies = {}

        # 1 & 2. Ingest Query & Language Detection
        t0 = time.time()
        detection = self.translator.detect(query)
        latencies["stage_1_2_detection_ms"] = round((time.time() - t0) * 1000.0, 2)

        # 3. Translate to MADRID Canonical Format
        t0 = time.time()
        madrid_enc = self.translator.encode(query, source_language_code=detection.language_code)
        latencies["stage_3_madrid_encode_ms"] = round((time.time() - t0) * 1000.0, 2)

        # 4 & 5. RAG Retrieval & Similarity Filtering
        t0 = time.time()
        # Search using MADRID canonical query for optimal retrieval accuracy
        search_query = madrid_enc.original_text if detection.language_code == "en" else madrid_enc.madrid_text
        results = self.retriever.retrieve(search_query, top_k=top_k)
        latencies["stage_4_5_retrieval_ms"] = round((time.time() - t0) * 1000.0, 2)

        # Build retrieved sources list & context
        retrieved_chunks = []
        sources_list = []
        context_parts = []
        max_similarity = 0.0

        for item in results:
            chunk, sim_score = item
            if sim_score > max_similarity:
                max_similarity = sim_score

            retrieved_chunks.append({
                "source": chunk["source"],
                "category": chunk.get("category", "general"),
                "topic": chunk.get("topic", "general"),
                "section": chunk.get("section", "Section"),
                "similarity_score": sim_score,
                "text": chunk["text"]
            })
            sources_list.append(chunk["source"])
            context_parts.append(f"Source: `{chunk['source']}` (Topic: {chunk.get('topic')})\n{chunk['text']}")

        sources_unique = list(dict.fromkeys(sources_list))
        context_str = "\n\n---\n\n".join(context_parts)

        # Grounding & Hallucination Guardrail Check
        if not results or max_similarity < similarity_threshold:
            fallback_answer = (
                "I could not find sufficient information in the QUANTLAB knowledge base to answer this query reliably."
            )
            decoded_fallback = self.translator.decode(fallback_answer, target_language_code=detection.language_code)
            
            token_stats = self.translator.get_token_breakdown(
                original_input=query,
                madrid_input=madrid_enc.madrid_text,
                rag_context="",
                llm_output_madrid=fallback_answer,
                final_output=decoded_fallback
            )
            
            return {
                "original_query": query,
                "detected_language": detection.language_name,
                "language_code": detection.language_code,
                "madrid_query": madrid_enc.madrid_text,
                "retrieved_sources": [],
                "madrid_response": fallback_answer,
                "final_answer": decoded_fallback,
                "token_stats": token_stats,
                "stage_latencies_ms": latencies,
                "total_latency_ms": round((time.time() - start_total) * 1000.0, 2),
                "status": "grounding_insufficient"
            }

        # Clean paragraph context synthesis from retrieved chunks
        narrative_context = clean_chunks_to_paragraphs(retrieved_chunks)

        # 6. LLM Synthesis (Output in MADRID representation & input language synthesis)
        t0 = time.time()
        prompt = (
            f"You are QUANTLAB Research Assistant.\n"
            f"Answer the user's question accurately using strictly the retrieved Knowledge Base Content and the User Query in MADRID canonical format.\n\n"
            f"USER QUERY IN MADRID:\n{madrid_enc.madrid_text}\n\n"
            f"RETRIEVED KNOWLEDGE BASE CONTENT:\n{context_str}\n\n"
            f"TARGET LANGUAGE: {detection.language_name} (Language Code: '{detection.language_code}')\n\n"
            f"RULES FOR YOUR RESPONSE:\n"
            f"1. Generate a complete, thorough, and well-structured answer in {detection.language_name} formatted in clear, proper paragraphs.\n"
            f"2. Explain the concept in detail using the retrieved knowledge base content.\n"
            f"3. STRICT TERMINOLOGY PRESERVATION: Do NOT translate or corrupt quantitative financial metrics and technical terms. Keep terms like Sharpe Ratio, CAGR, Maximum Drawdown, Sortino Ratio, Volatility, Slippage, Transaction Cost, SMA, EMA, RSI, OHLCV, NIFTY 50 exact.\n"
            f"4. Output a clean, professional, research-grade answer in proper narrative paragraphs without raw header tags or file paths.\n"
        )

        madrid_response = ""
        llm_direct_answer = ""

        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a quantitative research assistant. Answer accurately based on context in proper paragraphs."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                llm_direct_answer = res.choices[0].message.content.strip()
                madrid_response = f"[MADRID::RESPONSE | TOPIC:{retrieved_chunks[0]['topic']}]\n{llm_direct_answer}"
            except Exception:
                pass

        if not madrid_response:
            # Fallback grounded MADRID response using clean narrative context
            madrid_response = f"[MADRID::RESPONSE | TOPIC:{retrieved_chunks[0]['topic']}]\n{narrative_context}"

        latencies["stage_6_llm_synthesis_ms"] = round((time.time() - t0) * 1000.0, 2)

        # 7. Decode MADRID Response -> Original User Language in proper paragraphs
        t0 = time.time()
        if llm_direct_answer:
            final_answer = llm_direct_answer
        else:
            final_answer = self.translator.decode(madrid_response, target_language_code=detection.language_code)
        latencies["stage_7_madrid_decode_ms"] = round((time.time() - t0) * 1000.0, 2)

        # 8. Compute Token Accounting
        token_stats = self.translator.get_token_breakdown(
            original_input=query,
            madrid_input=madrid_enc.madrid_text,
            rag_context=context_str,
            llm_output_madrid=madrid_response,
            final_output=final_answer
        )

        total_latency = round((time.time() - start_total) * 1000.0, 2)

        return {
            "original_query": query,
            "detected_language": detection.language_name,
            "language_code": detection.language_code,
            "madrid_query": madrid_enc.madrid_text,
            "retrieved_sources": retrieved_chunks,
            "sources_summary": sources_unique,
            "madrid_response": madrid_response,
            "final_answer": final_answer,
            "token_stats": token_stats,
            "stage_latencies_ms": latencies,
            "total_latency_ms": total_latency,
            "status": "success"
        }

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Legacy interface wrapper for backward compatibility with server endpoints.
        """
        res = self.execute_pipeline(query, top_k=top_k)
        return {
            "query": query,
            "answer": res["final_answer"],
            "sources": res.get("sources_summary", []),
            "retrieved_chunks": res.get("retrieved_sources", []),
            "pipeline_data": res
        }
