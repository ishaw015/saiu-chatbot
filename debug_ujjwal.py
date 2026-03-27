from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

queries = [
    "Ujjwal",
    "Faculty: Ujjwal",
    "Faculty: Ujjwal course list semester credits",
    "Ujjwal teaches Programming in Python",
    "Programming in Python faculty Ujjwal",
    "what courses does Ujjwal teach this semester",
]

for q in queries:
    print(f"\n Query: '{q}'")
    results = vectorstore.similarity_search_with_score(
        q, k=3, filter={"source_type": "course_list_row"}
    )
    for doc, score in results:
        print(f"  Score: {score:.4f} | {doc.page_content[:120]}")