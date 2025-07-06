# 🧪 Methodology

The project follows a structured pipeline to build a **Retrieval-Augmented Generation (RAG)**-based chatbot capable of answering user questions using real consumer financial complaints from the CFPB dataset. The system leverages **semantic search** and **language models** to provide evidence-backed, context-aware responses.

---

### 1. Data Preparation and Cleaning

A dataset of consumer complaints was obtained and preprocessed for quality and relevance:

* **Filtered complaints** related to the following product categories:
    * Credit reporting
    * Personal loans
    * Buy Now, Pay Later (BNPL)
    * Savings accounts
    * Money transfers
* **Cleaned the Consumer complaint narrative field** by:
    * Lowercasing text
    * Removing special characters and numbers
    * Trimming extra whitespace
    * Removing null or incomplete entries
* **Exploratory Data Analysis (EDA)** was performed to understand data distribution, common issues, and narrative lengths, which informed downstream processing steps such as chunking.

---

### 2. Text Chunking and Embedding

To enhance retrieval accuracy, long narratives were split into smaller, coherent segments.

* **Text Chunking:**
    * Narratives were split using `RecursiveCharacterTextSplitter` with a **chunk size of 512 tokens** and an **overlap of 50 tokens**, ensuring continuity and contextual integrity.
* **Embedding Generation:**
    * Each chunk was converted into a dense vector representation using the **`sentence-transformers/all-MiniLM-L6-v2`** model, chosen for its efficiency and performance in generating high-quality sentence embeddings.

---

### 3. Vector Store Indexing

To enable fast and scalable similarity search over thousands of complaint chunks, a **FAISS index** was constructed:

* The index was built using **L2 (Euclidean) distance** for measuring similarity between vectors.
* Metadata (e.g., original text and source IDs) were stored separately and serialized using pickle.
* Both the FAISS index and metadata were persisted to disk for use in the retrieval phase.
* This step enabled efficient querying of relevant complaint excerpts during inference.

---

### 4. Retrieval-Augmented Generation (RAG)

The RAG pipeline integrates two key components: **retrieval** and **generation**.

* **Retrieval Phase:**
    * When a user inputs a query, it is embedded using the same model used during indexing. The FAISS index is queried to retrieve the most semantically similar complaint chunks.
* **Generation Phase:**
    * A question-answering model based on Hugging Face's transformer pipeline (`deepset/roberta-base-squad2`) was employed to generate concise and accurate answers using the retrieved chunks as context.
* **Prompt Engineering:**
    * A well-crafted prompt template ensured that:
        * Answers were grounded solely in the provided context
        * Responses remained clear, factual, and aligned with the user’s intent

---

### 5. Evaluation Strategy

A **qualitative evaluation approach** was adopted to assess the effectiveness of the RAG system.

* **Representative queries** covering common complaint themes were selected:
    * Incorrect information on reports
    * Delays in dispute resolution
    * Improper use of consumer reports
* For each query, the system’s output was manually reviewed to evaluate:
    * Relevance of the retrieved context
    * Accuracy and clarity of the generated response
    * Grounding in actual complaint data