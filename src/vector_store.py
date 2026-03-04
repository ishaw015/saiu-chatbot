import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, PDFMinerLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Setup Embeddings (Cached locally)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)

# Map filenames to categories for your app.py intent filtering
CATEGORY_MAP = {
    "Academic Calendar": "academic_calendar",
    "grades": "grading",
    "Student_Hand_Book": "policy",
    "IT_Policy": "policy",
    "Library Policy": "policy",
    "Course Catalogue": "semester_catalog",
    "Scholarship": "scholarship"
}

def build_vector_store():
    data_path = "data/"
    all_chunks = []
    
    # 2. Initialize Semantic Chunker
    # It splits text when it detects a "meaning shift" above the 90th percentile
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    print("[INFO] Processing documents...")
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        if file.endswith((".pdf", ".docx")):
            loader = PyPDFLoader(full_path) if file.endswith(".pdf") else Docx2txtLoader(full_path)
            docs = loader.load()
            
            # Determine Category
            category = "general"
            for key, val in CATEGORY_MAP.items():
                if key.lower() in file.lower():
                    category = val
            
            # Add metadata to chunks
            chunks = text_splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["filename"] = file
                chunk.metadata["doc_category"] = category
            
            all_chunks.extend(chunks)
            print(f" ✅ {file} -> {len(chunks)} semantic chunks")

    # 3. Create Vector Store
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="chroma_db/"
    )
    print(f"\n[INFO] ✅ Vector store built with {len(all_chunks)} chunks.")

if __name__ == "__main__":
    build_vector_store()