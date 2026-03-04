import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class Retriever:
    def __init__(self):
        # 1. Load the same embedding model used in vector_store.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./model_cache"
        )
        
        # 2. Load the existing vector store from the 'chroma_db' folder
        if os.path.exists("chroma_db"):
            self.vectorstore = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embeddings
            )
            print("[INFO] ✅ Retriever connected to ChromaDB.")
        else:
            print("[ERROR] ❌ chroma_db folder not found. Run vector_store.py first.")

    def get_relevant_chunks(self, query, k=5):
        """Basic retrieval used as a fallback or for general queries."""
        # similarity_search_with_score returns (Document, Score)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def format_context(self, results):
        """Formats the (doc, score) tuples into a string for the LLM."""
        context = ""
        for doc, score in results:
            src = doc.metadata.get("filename", "Unknown")
            cat = doc.metadata.get("doc_category", "General")
            context += f"\n[Source: {src} | Dept: {cat}]\n{doc.page_content}\n"
        return context

    def get_sources(self, results):
        """Extracts unique filenames from the results."""
        return list(set(doc.metadata.get("filename", "Unknown") for doc, score in results))