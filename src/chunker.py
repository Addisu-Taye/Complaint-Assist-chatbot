"""
# chunker.py
# Task: 2 - Text Chunking
# Created by: Addisu Taye
# Date: July 4, 2025
# Purpose: Split long narratives into smaller chunks for embedding
# Key Features: Uses RecursiveCharacterTextSplitter from LangChain
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)