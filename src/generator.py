"""
# generator.py
# Task: 3 - RAG Core Logic
# Created by: Addisu Taye
# Date: July 7, 2025
# Purpose: Use LLM to answer questions using retrieved context
# Key Features: QA pipeline using HuggingFace transformers
"""

from transformers import pipeline

class QAPipeline:
    def __init__(self):
        self.qa = pipeline("question-answering")

    def answer_question(self, question, context):
        result = self.qa(question=question, context=context)
        return result["answer"]