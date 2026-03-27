from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

results = vectorstore.similarity_search(
    "50% scholarship CGPA minimum maintain", k=5,
    filter={"doc_category": "scholarship"}
)

for i, doc in enumerate(results):
    print(f"--- Result {i+1} | Source: {doc.metadata.get('filename')} ---")
    print(doc.page_content[:600])
    print()