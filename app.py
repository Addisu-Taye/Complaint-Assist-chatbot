# app.py
# Intelligent Complaint Analyst for CrediTrust
# Built with LangChain, FAISS, and Gradio
# Created by: Addisu Taye
import os
import torch
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_rag_components():
    """
    Load all necessary components for the RAG system:
    - Embedding model
    - FAISS vector store
    - LLM and generation pipeline
    - Prompt template
    - RetrievalQA chain
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    # Load FAISS index
    index_path = "vector_store/faiss_index"
    vectorstore = FAISS.load_local(index_path, embeddings)

    # Load LLM components
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Create HF pipeline
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        min_length=50,
        do_sample=False,
        truncation=True,
        device=0 if device == "cuda" else -1
    )

    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Define prompt template
    prompt_template = """
    You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
    Use the following retrieved complaint excerpts to formulate your answer.
    If the context doesn't contain the answer, state that you don't have enough information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Build RAG chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    print("✅ RAG components loaded successfully")
    return rag_chain


def respond(question, rag_chain):
    """
    Handle user questions and return both the generated answer
    and the source documents used to generate it.
    """
    result = rag_chain({"query": question})
    answer = result['result']
    source_docs = result['source_documents']

    sources_text = ""
    for i, doc in enumerate(source_docs[:3]):
        sources_text += f"Source {i + 1}:\n"
        sources_text += f"Complaint ID: {doc.metadata.get('complaint_id', 'N/A')}\n"
        sources_text += f"Product: {doc.metadata.get('product', 'N/A')}\n"
        sources_text += f"Excerpt: {doc.page_content[:200]}...\n\n"

    return answer, sources_text


def create_interface(rag_chain):
    """
    Create the Gradio web interface for interactive querying.
    """
    def respond_wrapper(question, progress=gr.Progress()):
        progress(0.2, desc="Processing question...")
        answer, sources = respond(question, rag_chain)
        progress(0.8, desc="Formatting response...")
        return answer, sources

    with gr.Blocks(theme="soft", title="CrediTrust Complaint Analyst") as demo:
        gr.Markdown("## 🏦 CrediTrust Complaint Analyst\n### Ask questions about consumer complaints")

        with gr.Row():
            with gr.Column(scale=4):
                question = gr.Textbox(
                    label="Enter your question:",
                    placeholder="e.g., Why are customers complaining about credit cards?",
                    lines=3
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("🔍 Get Answer")

        with gr.Row():
            answer_box = gr.Textbox(label="AI Answer", lines=8, show_copy_button=True)

        with gr.Row():
            sources_box = gr.Textbox(label="Relevant Complaint Excerpts", lines=10)

        submit_btn.click(
            fn=respond_wrapper,
            inputs=question,
            outputs=[answer_box, sources_box]
        )

        # Example prompts
        examples = gr.Examples(
            examples=[
                ["Why are customers complaining about credit cards?"],
                ["What issues do people face with personal loans?"],
                ["What are common problems with Buy Now Pay Later services?"],
                ["Why are savings account holders unhappy?"],
                ["What complaints are common for money transfers?"]
            ],
            inputs=question
        )

    print("🎨 Interface created successfully")
    return demo


if __name__ == "__main__":
    print("🔄 Loading RAG components...")
    rag_chain = load_rag_components()

    print("🌐 Launching application...")
    demo = create_interface(rag_chain)

    print("🚀 Application ready! Opening interface...")
    demo.launch(share=True)