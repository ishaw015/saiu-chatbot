from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

results = vectorstore.similarity_search(
    "Programming in Python CS105 Ujjwal faculty semester", k=5,
    filter={"doc_category": "semester_catalog"}
)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:400])