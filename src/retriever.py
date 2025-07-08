"""
# retriever.py
# Task: 3 - RAG Core Logic
# Created by: Addisu Taye
# Date: July 7, 2025
# Purpose: Retrieve relevant complaint chunks based on user query
# Key Features: Uses FAISS for similarity search
"""

import faiss
import numpy as np
import pickle
from config import VECTOR_STORE_PATH

def load_index():
    index = faiss.read_index(VECTOR_STORE_PATH + ".index")
    with open(VECTOR_STORE_PATH + "_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve(query_embedding, index, metadata, top_k=5):
    D, I = index.search(np.array([query_embedding]), top_k)
    return [metadata[i] for i in I[0]]