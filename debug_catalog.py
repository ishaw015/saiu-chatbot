from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Check total catalog chunks and their metadata
results = vectorstore.get(where={"doc_category": "semester_catalog"})
print(f"Total catalog chunks: {len(results['ids'])}")

for i, metadata in enumerate(results['metadatas'][:10]):
    print(f"\nChunk {i+1}: {metadata}")