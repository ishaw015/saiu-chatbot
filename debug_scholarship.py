from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Check all scholarship chunks
results = vectorstore.get(where={"doc_category": "scholarship"})
print(f"Total scholarship chunks: {len(results['ids'])}")

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Metadata: {meta}")
    print(f"Content:\n{doc}")
    print("---")
    
    
# Add at the bottom of debug_scholarship.py
print("\n\n--- SIMILARITY SEARCH SCORES ---")
results2 = vectorstore.similarity_search_with_score(
    "50% scholarship CGPA minimum maintain 8", k=6,
    filter={"doc_category": "scholarship"}
)
for doc, score in results2:
    print(f"Score: {score:.4f} | Content: {doc.page_content[:150]}")