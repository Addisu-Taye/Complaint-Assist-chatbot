"""
# vector_store.py
# Task: 2 - Text Chunking, Embedding & Vector Store Indexing
# Created by: Addisu Taye
# Date: July 5, 2025
# Purpose: Build and persist FAISS index using embeddings of complaint narratives
# Key Features:
- Builds FAISS index from embeddings
- Saves index and metadata for later use
- Supports L2 distance (Euclidean) for similarity search
"""

import faiss
import numpy as np
import pickle
import os

from config import VECTOR_STORE_PATH


def build_index(embeddings, dimension=None, save=True):
    """
    Builds a FAISS index from given embeddings.
    
    Args:
        embeddings (np.ndarray or list): List or array of embedding vectors.
        dimension (int): Dimension of each embedding vector (optional).
        save (bool): Whether to save the index to disk.

    Returns:
        faiss.Index: The built FAISS index
    """
    # Convert to numpy array if needed
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    if dimension is None:
        dimension = embeddings.shape[1]

    # Create a flat (brute-force) L2 distance index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors")

    if save:
        # Ensure directory exists
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)

        # Save index to disk
        faiss.write_index(index, VECTOR_STORE_PATH + ".index")
        print(f"💾 FAISS index saved to {VECTOR_STORE_PATH}.index")

    return index


def save_metadata(metadata, file_path=None):
    """
    Saves metadata (like source text or IDs) to disk.
    
    Args:
        metadata (list): List of dictionaries or strings containing metadata.
        file_path (str): Optional custom path to save metadata
    """
    if file_path is None:
        file_path = VECTOR_STORE_PATH + "_metadata.pkl"

    with open(file_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"💾 Metadata saved to {file_path}")