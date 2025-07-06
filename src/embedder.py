"""
# embedder.py
# Task: 2 - Embedding Generation
# Created by: Addisu Taye
# Date: July 4, 2025
# Purpose: Generate embeddings using Sentence Transformers
# Key Features: Embed text using all-MiniLM-L6-v2 model
"""

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name=None):
        self.model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")

    def embed(self, texts):
        return self.model.encode(texts)