import requests
import csv
import time

# Configuration
API_URL = "http://localhost:8000/chat"
TEST_CASES = [
    "Which faculty is teaching Programming in Python?",
    "What is the deadline to add or drop a course this semester?",
    "What CGPA do I need to maintain if I have a 50% scholarship?",
    "What courses are offered in SCDS this semester?",
    "what courses does ujjwal teach this semester?",
    "What are the holidays in 2026?",
    "Can I audit a course this semester?",
    "What is the fee structure for the BBA program?",
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

            if "detail" in data:
                results.append({
                    "query": query,
                    "answer": f"API ERROR: {data['detail']}",
                    "sources": "",
                    "latency": latency,
                    "status": "❌ API Error"
                })
                continue

            results.append({
                "query": query,
                "answer": data.get("answer", "No answer"),
                "sources": ", ".join(data.get("sources", [])),
                "latency": latency,
                "status": "✅ Success" if data.get("sources") else "⚠️ No Sources"
            })
        except Exception as e:
            results.append({
                "query": query,
                "answer": str(e),
                "sources": "",
                "latency": round(time.time() - start_time, 2),
                "status": "❌ Failed"
            })

        time.sleep(15)  # ← AFTER request, outside try/except, gap between queries

    # Save to CSV
    with open("rag_test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "status", "latency", "sources", "answer"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n✅ Test complete! Results saved to 'rag_test_results.csv'")

if __name__ == "__main__":
    run_tests()