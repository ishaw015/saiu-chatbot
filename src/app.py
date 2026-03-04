from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from src.retriever import Retriever
from dotenv import load_dotenv
import os

# ── Load environment variables ─────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Initialize FastAPI ─────────────────────────────────────
app = FastAPI(
    title="SaiU Academic Chatbot",
    description="RAG-powered chatbot for Sai University",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Initializing retriever...")
retriever = Retriever()
groq_client = Groq(api_key=GROQ_API_KEY)
print("[INFO] ✅ App ready.")


# ────────────────────────────────────────────────────────────
# Request / Response Models
# ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    chat_history: list = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ────────────────────────────────────────────────────────────
# System Prompt
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the SaiU Assistant, an official academic advisor for Sai University.
You have access to verified university documents through the context provided below.

### OPERATIONAL RULES:
1. GROUNDING: Answer ONLY using the information in the [Source] headers and content provided. 
2. CITATION: Every claim you make must be followed by a source reference in square brackets, e.g., "The attendance requirement is 75% [School_of_Law.pdf]".
3. AMBIGUITY: If the context mentions different rules for different schools, clarify this for the user (e.g., "In the School of Computing, X applies, while in the School of Law, Y applies").
4. FALLBACK: If the answer is not in the context, say EXACTLY: 
   "I'm sorry, I don't have that specific information in my current database. Please contact the Registrar's Office at registrar@saiuniversity.edu.in for official clarification."
5. TONE: Professional, supportive, and concise. Use bullet points for complex lists.

### CONTEXT FORMAT:
The context provided will look like this:
[Source: filename.pdf | Dept: category]
<content of the document>
"""


# ────────────────────────────────────────────────────────────
# INTENT DETECTION
# ────────────────────────────────────────────────────────────
def detect_intent(question: str) -> str:
    q = question.lower()

    # Prioritize Semester Catalog for "this semester" or "offered" questions
    if any(k in q for k in ["this semester", "current semester", "2025-26", "offered", "catalog"]):
        return "semester"

    # Priority for Grading/SGPA
    if any(k in q for k in ["cgpa", "sgpa", "grading", "calculate", "grade point"]):
        return "grading"

    if any(k in q for k in ["credits", "total credits", "graduation requirement", "how many credits"]):
        return "structure"

    if any(k in q for k in ["courses offered", "subjects", "major courses", "curriculum", "specializations"]):
        return "curriculum"

    if any(k in q for k in ["exam", "holiday", "semester start", "important dates"]):
        return "calendar"

    if any(k in q for k in ["timetable", "schedule", "room number"]):
        return "timetable"

    if "scholarship" in q:
        return "scholarship"

    if any(k in q for k in ["code of conduct", "policy", "misconduct", "library rules", "it policy"]):
        return "policy"

    return "general"


# ────────────────────────────────────────────────────────────
# SCHOOL DETECTION
# ────────────────────────────────────────────────────────────
def detect_school(question: str):
    q = question.lower()

    school_map = {
        "computing": "school_computing_datascience.pdf",
        "data science": "school_computing_datascience.pdf",
        "business": "school_business.pdf",
        "artificial intelligence": "school_artificial_intelligence.pdf",
        "law": "school_law.pdf",
        "media": "school_media.pdf",
        "arts": "school-arts_sciences.pdf",
        "technology": "school-technology.pdf"
    }

    for keyword, filename in school_map.items():
        if keyword in q:
            return filename

    return None


# ────────────────────────────────────────────────────────────
# DYNAMIC SECTION MAPPING (NOT HARDCODED PER QUERY)
# ────────────────────────────────────────────────────────────
INTENT_SECTION_MAP = {
    "structure": ["credit_structure", "eligibility", "program_structure"],
    "curriculum": ["key_courses", "core_courses", "overview"]
}

# ────────────────────────────────────────────────────────────
# UPDATED RETRIEVAL ENGINE WITH SAFETY NET
# ────────────────────────────────────────────────────────────
def retrieve_by_intent(intent: str, question: str):
    target_school_file = detect_school(question)
    all_candidates = []

    # A. Targeted Intent Search (Using .extend to prevent overwriting)
    if intent == "semester":
        # If a school is mentioned, prioritize catalog chunks for that school
        search_filter = {"doc_category": "semester_catalog"}
        if target_school_file:
             # This helps find the Business section inside the big catalog
             all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
                question, k=15, filter=search_filter
            ))
        else:
            all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
                question, k=12, filter=search_filter
            ))
            
    elif intent == "grading":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"doc_category": "grading"}
        ))
    elif intent == "structure" and target_school_file:
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question + " total credits degree requirements", k=15, filter={"filename": target_school_file}
        ))
    elif intent == "scholarship":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"doc_category": "scholarship"}
        ))
    elif intent == "policy":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"doc_category": "policy"}
        ))

    # B. School-Specific Search (Layer 2)
    if target_school_file:
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"filename": target_school_file}
        ))

    # C. Global Fallback (Layer 3 - The Safety Net)
    all_candidates.extend(retriever.get_relevant_chunks(question, k=10))

    # D. DE-DUPLICATION (Crucial!)
    # We use the page_content or a unique ID to ensure we don't send duplicates to the LLM
    unique_results = []
    seen_contents = set()
    
    for doc, score in all_candidates:
        if doc.page_content not in seen_contents:
            unique_results.append((doc, score))
            seen_contents.add(doc.page_content)

    # E. Sort by Score (Lowest distance first)
    unique_results.sort(key=lambda x: x[1])
    
    return unique_results


# ────────────────────────────────────────────────────────────
# IMPROVED CHAT ENDPOINT WITH SCORE FILTERING
# ────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        intent = detect_intent(request.question)
        raw_results = retrieve_by_intent(intent, request.question)
        
        # 1. SCORE FILTERING (Strictness against hallucinations)
        # Chroma scores: lower is better. 0.6 is a safe "relevant" threshold.
        filtered_results = []
        for doc, score in raw_results:
            if score < 0.95:  # Adjust this based on your stress_tests
                filtered_results.append((doc, score))

        # 2. De-duplicate snippets
        unique_chunks = []
        seen_text = set()
        for doc, score in filtered_results:
            snippet = doc.page_content[:150].strip()
            if snippet not in seen_text:
                unique_chunks.append((doc, score))
                seen_text.add(snippet)

        final_selection = unique_chunks[:12] # Send top 8 clear chunks

        # 3. Handle "No Context Found" scenario early
        if not final_selection:
            return ChatResponse(
                answer="I'm sorry, I don't have that specific information in my current database. Please contact the Registrar's Office at registrar@saiuniversity.edu.in for official clarification.",
                sources=[]
            )

        # 4. Format for LLM
        context = ""
        sources = []
        for doc, _ in final_selection:
            src = doc.metadata.get("filename", "Unknown")
            cat = doc.metadata.get("doc_category", "General")
            context += f"\n[Source: {src} | Dept: {cat}]\n{doc.page_content}\n"
            if src not in sources:
                sources.append(src)

        # 5. Generate with Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ],
            temperature=0.1, # Reduced for even less hallucination
            max_tokens=1024
        )

        answer = response.choices[0].message.content
        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
# ────────────────────────────────────────────────────────────
# Health Check
# ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "message": "SaiU Chatbot is running!"}


# ────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)