"""
Web app entry point for the RAG-based medical chatbot.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from chatbot import MedicalChatbot


def load_bot() -> MedicalChatbot:
    """
    Creates the chatbot once and reuses it across reruns.
    """
    if "bot" not in st.session_state:
        st.session_state.bot = MedicalChatbot()
    return st.session_state.bot


def main() -> None:
    """
    Streamlit web app.
    """
    st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺", layout="wide")

    st.title("RAG-based Medical Chatbot for Symptom Analysis")
    st.caption("A simple semester project using Retrieval-Augmented Generation, FAISS, and Sentence Transformers.")

    with st.sidebar:
        st.subheader("About This Project")
        st.write("Pipeline: Documents -> Chunking -> Embeddings -> FAISS -> Retrieval -> LLM -> Answer")
        st.write("The chatbot answers only from the retrieved medical dataset.")
        st.warning("This app is for educational use only.")

        st.subheader("Example Symptoms")
        st.write("- fever, body pain, chills")
        st.write("- headache, nausea, light sensitivity")
        st.write("- vomiting, stomach pain, diarrhea")

    bot = load_bot()

    st.subheader("Enter Symptoms")
    user_input = st.text_area(
        "Describe your symptoms",
        placeholder="Example: fever, cough, fatigue, sore throat",
        height=140,
    )

    analyze_clicked = st.button("Analyze Symptoms", use_container_width=True)

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please enter symptoms before analyzing.")
            return

        result = bot.analyze_symptoms(user_input)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Possible Condition")
            st.info(result["possible_condition"])

            st.subheader("Suggested Remedy")
            st.success(result["suggested_remedy"])

        with col2:
            st.subheader("Disclaimer")
            st.warning(result["disclaimer"])

        if result["retrieved_chunks"]:
            with st.expander("Show Retrieved Medical Context"):
                for index, chunk in enumerate(result["retrieved_chunks"], start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(f"Condition: {chunk['condition']}")
                    st.write(f"Similarity score: {chunk['score']}")
                    st.write(chunk["text"])
                    st.divider()


if __name__ == "__main__":
    main()
