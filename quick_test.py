from src.retriever import Retriever

def test_retrieval():
    # Initialize the updated retriever
    retriever = Retriever()
    
    # The query we want to check
    query = "When are the end term exams for semester 2?"
    
    print(f"\n🔍 Testing Query: {query}")
    
    # Accessing the vectorstore directly from the retriever
    # In the new retriever.py, it is self.vectorstore (not self.vs)
    results = retriever.vectorstore.similarity_search_with_score(query, k=3)
    
    if not results:
        print("❌ No results found in the vector store.")
        return

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {score:.4f} (Lower is better)")
        print(f"Source: {doc.metadata.get('filename')}")
        print(f"Category: {doc.metadata.get('doc_category')}")
        print(f"Snippet: {doc.page_content[:200]}...")
    print("\n" + "="*50)

if __name__ == "__main__":
    test_retrieval()