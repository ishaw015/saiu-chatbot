from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import re

def advanced_semantic_chunking(documents):
    # 1. Use a stronger embedding model for the splitting logic
    # BGE models are excellent for detecting semantic boundaries
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Initialize the Semantic Chunker
    # It calculates the 'cosine distance' between sentences and splits where meaning shifts
    semantic_splitter = SemanticChunker(
        embed_model, 
        breakpoint_threshold_type="percentile", # Usually 90-95% is the sweet spot
        breakpoint_threshold_amount=95
    )
    
    final_chunks = []
    
    for doc in documents:
        # CLEANING: Keep numbers, remove excessive whitespace
        clean_content = re.sub(r'\s+', ' ', doc.page_content).strip()
        
        # 3. Contextual Injection (The "Secret Sauce")
        # We create a header that tells the Vector Store exactly what this file is about.
        file_name = doc.metadata.get('filename', 'Unknown Document')
        category = doc.metadata.get('doc_category', 'General')
        header = f"[Source: {file_name} | Dept: {category}]\n"
        
        # Split the document into semantic segments
        chunks = semantic_splitter.split_text(clean_content)
        
        for content in chunks:
            # We prepend the header to the text so the VECTOR captures the context
            # This prevents "orphan chunks" that have no context.
            contextualized_text = header + content
            
            # Create the final Document object
            final_chunks.append(Document(
                page_content=contextualized_text,
                metadata=doc.metadata
            ))
            
    print(f"[INFO] Created {len(final_chunks)} semantic chunks from {len(documents)} docs.")
    return final_chunks