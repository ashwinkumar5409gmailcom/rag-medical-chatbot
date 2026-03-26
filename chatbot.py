"""
Main chatbot logic:
1. Checks safety keywords
2. Retrieves medical context
3. Generates a grounded answer using OpenAI or a mock generator
"""

from __future__ import annotations

import os
from typing import Dict, List

from data_loader import chunk_documents, load_medical_documents
from embedder import MedicalEmbedder
from retriever import MedicalRetriever
from vector_store import FAISSVectorStore

DISCLAIMER = "This is not a medical diagnosis."
EMERGENCY_KEYWORDS = ["chest pain", "breathing difficulty"]


class MedicalChatbot:
    """
    End-to-end RAG chatbot for simple symptom analysis.
    """

    def __init__(self) -> None:
        documents = load_medical_documents()
        chunks = chunk_documents(documents)

        self.embedder = MedicalEmbedder()
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode_texts(chunk_texts)

        self.vector_store = FAISSVectorStore(embedding_dimension=embeddings.shape[1])
        self.vector_store.add_documents(embeddings, chunks)

        self.retriever = MedicalRetriever(self.embedder, self.vector_store)

    def check_emergency(self, user_input: str) -> bool:
        """
        Returns True if emergency symptoms are found.
        """
        lowered_input = user_input.lower()
        return any(keyword in lowered_input for keyword in EMERGENCY_KEYWORDS)

    def generate_response(self, user_input: str, top_k: int = 3) -> str:
        """
        Generates a safe answer using retrieved context only.
        """
        result = self.analyze_symptoms(user_input, top_k=top_k)
        return (
            f"Possible condition: {result['possible_condition']}\n"
            f"Suggested remedy: {result['suggested_remedy']}\n"
            f"Disclaimer: {result['disclaimer']}"
        )

    def analyze_symptoms(self, user_input: str, top_k: int = 3) -> Dict[str, object]:
        """
        Returns a structured result that is easier to show in a web app.
        """
        if self.check_emergency(user_input):
            return {
                "possible_condition": "Emergency symptom detected",
                "suggested_remedy": "Please seek immediate medical attention or contact emergency services.",
                "disclaimer": DISCLAIMER,
                "retrieved_chunks": [],
            }

        retrieved_results = self.retriever.retrieve(user_input, top_k=top_k)

        if not retrieved_results:
            return {
                "possible_condition": "No relevant condition found",
                "suggested_remedy": "Please consult a qualified doctor for proper evaluation.",
                "disclaimer": DISCLAIMER,
                "retrieved_chunks": [],
            }

        # If similarity is too low, we avoid making a strong guess.
        best_score = retrieved_results[0]["score"]
        if best_score < 0.25:
            return {
                "possible_condition": "The symptoms do not clearly match the available dataset",
                "suggested_remedy": "Monitor your symptoms, rest, stay hydrated, and consult a doctor if they continue.",
                "disclaimer": DISCLAIMER,
                "retrieved_chunks": [
                    {
                        "condition": result["document"]["condition"],
                        "text": result["document"]["text"],
                        "score": round(result["score"], 3),
                    }
                    for result in retrieved_results
                ],
            }

        context = self._build_context(retrieved_results)
        llm_response = self._call_openai(user_input, context)

        if llm_response is None:
            llm_response = self._mock_llm_response(retrieved_results)

        return {
            "possible_condition": llm_response["possible_condition"],
            "suggested_remedy": llm_response["suggested_remedy"],
            "disclaimer": llm_response["disclaimer"],
            "retrieved_chunks": [
                {
                    "condition": result["document"]["condition"],
                    "text": result["document"]["text"],
                    "score": round(result["score"], 3),
                }
                for result in retrieved_results
            ],
        }

    def _build_context(self, retrieved_results: List[Dict[str, object]]) -> str:
        """
        Combines retrieved chunks into one context block for the LLM.
        """
        context_parts = []
        for result in retrieved_results:
            doc = result["document"]
            context_parts.append(doc["text"])
        return "\n".join(context_parts)

    def _call_openai(self, user_input: str, context: str) -> Dict[str, str] | None:
        """
        Uses OpenAI only if the package and API key are available.
        The prompt explicitly restricts the answer to retrieved context.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                "You are a medical support chatbot for a college project.\n"
                "Answer ONLY from the retrieved context.\n"
                "Do not add outside medical knowledge.\n"
                "If the context is insufficient, say that the dataset is insufficient.\n"
                "Return exactly in this format:\n"
                "Possible condition: ...\n"
                "Suggested remedy: ...\n"
                f"Disclaimer: {DISCLAIMER}\n\n"
                f"User symptoms: {user_input}\n\n"
                f"Retrieved context:\n{context}"
            )

            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
            )
            return self._parse_response_text(response.output_text.strip())
        except Exception:
            return None

    def _mock_llm_response(self, retrieved_results: List[Dict[str, object]]) -> Dict[str, str]:
        """
        Fallback response generator when OpenAI API is not available.
        It still uses only retrieved context.
        """
        top_document = retrieved_results[0]["document"]
        top_condition = top_document["condition"]
        top_text = top_document["text"]

        remedy_text = "Please consult a doctor for guidance."
        if "Basic remedies:" in top_text:
            remedy_text = top_text.split("Basic remedies:", maxsplit=1)[1].strip()

        return {
            "possible_condition": top_condition,
            "suggested_remedy": remedy_text,
            "disclaimer": DISCLAIMER,
        }

    def _parse_response_text(self, response_text: str) -> Dict[str, str]:
        """
        Converts LLM text output into a structured dictionary.
        """
        parsed = {
            "possible_condition": "Dataset is insufficient",
            "suggested_remedy": "Please consult a qualified doctor for proper evaluation.",
            "disclaimer": DISCLAIMER,
        }

        for line in response_text.splitlines():
            cleaned_line = line.strip()
            if cleaned_line.lower().startswith("possible condition:"):
                parsed["possible_condition"] = cleaned_line.split(":", maxsplit=1)[1].strip()
            elif cleaned_line.lower().startswith("suggested remedy:"):
                parsed["suggested_remedy"] = cleaned_line.split(":", maxsplit=1)[1].strip()
            elif cleaned_line.lower().startswith("disclaimer:"):
                parsed["disclaimer"] = cleaned_line.split(":", maxsplit=1)[1].strip()

        return parsed
