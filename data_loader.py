"""
Loads a small medical knowledge base and converts it into RAG-ready chunks.
"""

from __future__ import annotations

from typing import Dict, List


# Small beginner-friendly dataset for a semester project.
MEDICAL_DATA: List[Dict[str, object]] = [
    {
        "condition": "Common Cold",
        "symptoms": ["runny nose", "sneezing", "sore throat", "mild cough"],
        "causes": "Usually caused by a viral infection that affects the upper respiratory tract.",
        "remedies": "Drink warm fluids, take proper rest, use steam inhalation, and stay hydrated.",
    },
    {
        "condition": "Flu",
        "symptoms": ["fever", "body pain", "fatigue", "cough", "chills"],
        "causes": "Caused by the influenza virus and spreads from person to person.",
        "remedies": "Rest well, drink fluids, eat light meals, and monitor high fever.",
    },
    {
        "condition": "Migraine",
        "symptoms": ["headache", "nausea", "light sensitivity", "throbbing pain"],
        "causes": "Can be triggered by stress, poor sleep, certain foods, or hormonal changes.",
        "remedies": "Rest in a dark quiet room, stay hydrated, and avoid known triggers.",
    },
    {
        "condition": "Acidity",
        "symptoms": ["burning chest sensation", "indigestion", "bloating", "sour taste"],
        "causes": "Often caused by spicy food, overeating, or lying down after meals.",
        "remedies": "Eat smaller meals, avoid spicy food, and do not lie down right after eating.",
    },
    {
        "condition": "Food Poisoning",
        "symptoms": ["vomiting", "diarrhea", "stomach pain", "nausea"],
        "causes": "Usually caused by contaminated food or water containing harmful germs.",
        "remedies": "Drink plenty of fluids, rest, and avoid oily or heavy food.",
    },
    {
        "condition": "Allergy",
        "symptoms": ["sneezing", "itching", "rash", "watery eyes"],
        "causes": "Triggered by allergens such as dust, pollen, certain foods, or pets.",
        "remedies": "Avoid the trigger, keep surroundings clean, and use soothing care for rash or irritation.",
    },
    {
        "condition": "Asthma",
        "symptoms": ["wheezing", "cough", "shortness of breath", "chest tightness"],
        "causes": "Can be triggered by dust, smoke, exercise, cold air, or allergies.",
        "remedies": "Avoid triggers, sit upright, stay calm, and use prescribed inhaler support if available.",
    },
    {
        "condition": "Bronchitis",
        "symptoms": ["persistent cough", "mucus", "fatigue", "mild fever"],
        "causes": "Often caused by viral infection or irritation in the bronchial tubes.",
        "remedies": "Rest, warm fluids, steam inhalation, and avoid smoke exposure.",
    },
    {
        "condition": "Diabetes",
        "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
        "causes": "Happens when the body does not make enough insulin or cannot use it properly.",
        "remedies": "Limit sugar intake, maintain regular exercise, and follow medical advice for glucose control.",
    },
    {
        "condition": "Hypertension",
        "symptoms": ["headache", "dizziness", "fatigue", "blurred vision"],
        "causes": "Linked with stress, high salt intake, obesity, lack of exercise, or genetics.",
        "remedies": "Reduce salt, manage stress, exercise regularly, and monitor blood pressure.",
    },
    {
        "condition": "Anemia",
        "symptoms": ["fatigue", "weakness", "pale skin", "dizziness"],
        "causes": "Commonly caused by low iron levels, blood loss, or poor nutrition.",
        "remedies": "Eat iron-rich foods such as spinach and legumes, and discuss supplements with a doctor.",
    },
    {
        "condition": "Dehydration",
        "symptoms": ["dry mouth", "dizziness", "fatigue", "dark urine"],
        "causes": "Occurs when the body loses more fluids than it takes in.",
        "remedies": "Drink water or oral rehydration fluids and rest in a cool place.",
    },
]


def load_medical_documents() -> List[Dict[str, str]]:
    """
    Converts the dataset into plain text documents.
    Each document represents one disease/condition.
    """
    documents: List[Dict[str, str]] = []

    for item in MEDICAL_DATA:
        text = (
            f"Condition: {item['condition']}. "
            f"Symptoms: {', '.join(item['symptoms'])}. "
            f"Causes: {item['causes']} "
            f"Basic remedies: {item['remedies']}"
        )

        documents.append(
            {
                "id": item["condition"].lower().replace(" ", "_"),
                "text": text,
                "condition": str(item["condition"]),
            }
        )

    return documents


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 45,
    overlap: int = 10,
) -> List[Dict[str, str]]:
    """
    Splits each document into smaller chunks.
    Word-based chunking keeps the logic easy to understand for students.
    """
    chunks: List[Dict[str, str]] = []

    for document in documents:
        words = document["text"].split()
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append(
                {
                    "chunk_id": f"{document['id']}_chunk_{chunk_index}",
                    "condition": document["condition"],
                    "text": chunk_text,
                }
            )

            if end == len(words):
                break

            start = end - overlap
            chunk_index += 1

    return chunks
