import requests
import csv
import time

# Configuration
API_URL = "http://localhost:8000/chat"
TEST_CASES = [
    # 1. School-Specific (Checks if School Detection works)
    "What are the core courses for the School of Computing and Data Sciences?",
    "Tell me about the curriculum for the School of Law.",
    "What major courses are offered in School of Business this semester?",
    
    # 2. Policy & Grading (Checks Intent Detection)
    "What is the minimum CGPA required to graduate?",
    "How is the SGPA calculated at SaiU?",
    "What is the attendance policy for final exams?",
    
    # 3. Scholarships (Checks Document Classification)
    "Are there any merit-based scholarships available?",
    "What is the deadline for scholarship renewal?",
    
    # 4. Out-of-Context (Checks the Fallback "Registrar" message)
    "What is the menu in the university cafeteria today?",
    "How do I apply for a parking permit?",
    
    # 5. Intent Overlap
    "How many credits do I need for Law and does it affect my scholarship?",
    "How many total credits are needed to graduate from school of computing and Data Science?"
]

def run_tests():
    results = []
    print(f"🚀 Starting Stress Test on {len(TEST_CASES)} queries...\n")

    for query in TEST_CASES:
        print(f"🔍 Testing: {query}")
        start_time = time.time()
        
        try:
            response = requests.post(API_URL, json={"question": query})
            data = response.json()
            latency = round(time.time() - start_time, 2)
            
            results.append({
                "query": query,
                "answer": data["answer"],
                "sources": ", ".join(data["sources"]),
                "latency": latency,
                "status": "✅ Success" if len(data["sources"]) > 0 else "⚠️ No Sources"
            })
        except Exception as e:
            results.append({"query": query, "answer": str(e), "sources": "", "status": "❌ Failed"})

    # Save to CSV
    with open("rag_test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "status", "latency", "sources", "answer"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n✅ Test complete! Results saved to 'rag_test_results.csv'")

if __name__ == "__main__":
    run_tests()