"""
VectorForge — Gemini LLM Service for RAG (Retrieval-Augmented Generation)
"""

from __future__ import annotations

import os
import time
import httpx
from typing import Optional, Any


class GeminiService:
    """Service to handle RAG prompt synthesis with Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def construct_rag_prompt(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Format grounded RAG prompt from retrieved context documents."""
        system_instruction = (
            system_prompt
            or "You are an intelligent assistant integrated with VectorForge, a custom high-performance vector database. "
               "Answer the user's query based strictly on the provided context retrieved from the vector database. "
               "Be clear, accurate, and concise. If the context is limited, use your domain knowledge to provide a helpful response."
        )

        context_str = "\n".join(
            [f"- [{ctx.get('id', 'item')}] (similarity: {ctx.get('similarity', 0.0):.4f}): {ctx.get('text', '')}"
             for ctx in contexts]
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"--- RETRIEVED CONTEXT FROM VECTOR DB ---\n"
            f"{context_str if context_str else 'No relevant context found.'}\n"
            f"----------------─────────────────────────\n\n"
            f"USER QUERY: {query}\n\n"
            f"ANSWER:"
        )
        return prompt

    async def generate_rag_response(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> tuple[str, float, str]:
        """
        Generate answer using Gemini LLM.
        Returns: (answer_text, latency_ms, model_name)
        """
        start_time = time.perf_counter()
        prompt = self.construct_rag_prompt(query, contexts, system_prompt)

        # Fallback mode if GEMINI_API_KEY is not set
        if not self.api_key:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            top_texts = [ctx.get("text", "") for ctx in contexts[:2]]
            fallback_answer = (
                f"[VectorForge RAG Engine]: GEMINI_API_KEY not configured in environment. "
                f"Retrieved {len(contexts)} top matching context(s) from VectorForge index.\n\n"
                f"Top Context: '{top_texts[0]}' if top_texts else 'None'"
            )
            return fallback_answer, elapsed_ms, f"{self.model}-simulated"

        # Execute live request to Google Gemini API
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024
            }
        }

        url = f"{self.base_url}?key={self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                # Extract response text
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        answer = parts[0].get("text", "").strip()
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return answer, elapsed_ms, self.model

                answer = "No response text received from Gemini API."
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return answer, elapsed_ms, self.model

        except Exception as err:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_answer = (
                f"[VectorForge RAG Engine]: Error communicating with Gemini API ({type(err).__name__}: {str(err)}). "
                f"Retrieved {len(contexts)} source contexts successfully."
            )
            return error_answer, elapsed_ms, self.model


# Global singleton instance
gemini_service = GeminiService()
