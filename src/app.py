from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from src.retriever import Retriever
from dotenv import load_dotenv
import os

# ── Load environment variables ─────────────────────────────
load_dotenv()
PRIMARY_KEY = os.getenv("GROQ_API_KEY")
FALLBACK_KEY = os.getenv("GROQ_API_KEY_2")

# ── Initialize FastAPI ─────────────────────────────────────
app = FastAPI(
    title="SaiU Academic Chatbot",
    description="RAG-powered chatbot for Sai University",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://famous-banoffee-110f75.netlify.app", "https://fanciful-entremet-b10e21.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Initializing retriever...")
retriever = Retriever()
groq_client = Groq(api_key=PRIMARY_KEY)
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
3. SEMESTER COURSES: When a question asks about courses offered THIS semester or current courses:
   - Use ONLY [Source: Course Catalogue_25-26_Sem2.docx] for the course list.
   - Do NOT mix in program names or degree titles from school overview files as if they are courses.
   - School overview files describe degree programs, NOT individual semester courses.
   - If the catalogue lists specific course names, list those. If not found, trigger rule 5.
   - Do NOT cite school overview files in the final note or disclaimer.
4. AMBIGUITY: If the context mentions different rules for different schools, clarify this for the user (e.g., "In the School of Computing, X applies, while in the School of Law, Y applies").
5. FALLBACK: If the answer is not in the context, say EXACTLY:
   "I'm sorry, I don't have that specific information in my current database. Please contact the Registrar's Office at registrar@saiuniversity.edu.in for official clarification."
6. TONE: Professional, supportive, and concise. Use bullet points for complex lists.

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

    if any(k in q for k in ["this semester", "current semester", "2025-26", "offered",
                              "catalog", "add or drop", "withdraw", "audit",
                              "minimum credits", "credit requirement",
                              "faculty", "instructor", "teaching", "who teaches",
                              "who is taking", "professor", "course code",
                              "how many credits is", "credits is",
                              "prerequisites", "syllabus", "learning outcomes",
                              "what courses does", "which courses does",
                              "courses taught by", "what does",
                              "courses does"]):
        return "semester"

    if any(k in q for k in ["cgpa", "sgpa", "grading", "calculate", "grade point"]):
        return "grading"

    if any(k in q for k in ["gpa", "grade"]) and "scholarship" not in q:
        return "grading"

    if any(k in q for k in ["credits", "total credits", "graduation requirement", "how many credits"]):
        return "structure"

    if any(k in q for k in ["courses offered", "subjects", "major courses", "curriculum", "specializations"]):
        return "curriculum"

    if any(k in q for k in ["exam", "holiday", "holidays", "holiday list",
                              "semester start", "important dates", "when is"]):
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
        "technology": "school-technology.pdf",
        "scds": "school_computing_datascience.pdf",
        "sol": "school_law.pdf",
        "sob": "school_business.pdf",
        "soai": "school_artificial_intelligence.pdf",
        "sas": "school-arts_sciences.pdf",
        "sot": "school-technology.pdf",
    }

    for keyword, filename in school_map.items():
        if keyword in q:
            return filename

    return None


# ────────────────────────────────────────────────────────────
# DYNAMIC SECTION MAPPING
# ────────────────────────────────────────────────────────────
INTENT_SECTION_MAP = {
    "structure": ["credit_structure", "eligibility", "program_structure"],
    "curriculum": ["key_courses", "core_courses", "overview"]
}


# ────────────────────────────────────────────────────────────
# FACULTY NAME SEARCH (bypasses embedding for proper nouns)
# ────────────────────────────────────────────────────────────
def search_faculty_by_name(name: str) -> list:
    """
    Directly scan course_list_row chunks for a faculty name.
    Bypasses vector search which fails for proper nouns.
    """
    all_rows = retriever.vectorstore.get(
        where={"source_type": "course_list_row"}
    )
    matches = []
    for doc_text, meta in zip(all_rows["documents"], all_rows["metadatas"]):
        if name.lower() in doc_text.lower():
            matches.append((
                type('Document', (), {'page_content': doc_text, 'metadata': meta})(),
                0.1  # low score so these rank first
            ))
    return matches


# ────────────────────────────────────────────────────────────
# RETRIEVAL ENGINE
# ────────────────────────────────────────────────────────────
def retrieve_by_intent(intent: str, question: str):
    target_school_file = detect_school(question)
    all_candidates = []

    # A. Targeted Intent Search
    if intent == "semester":
        catalog_query = question + " course name credits description syllabus outline faculty instructor"

        if target_school_file:
            school_tag = target_school_file.replace(".pdf", "")
            all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
                catalog_query, k=15, filter={"school_tag": school_tag}
            ))
            all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
                question, k=5, filter={"filename": target_school_file}
            ))

        # Faculty name search via vector similarity
        skip_words = {"what", "which", "courses", "does", "teach", "this",
                      "semester", "who", "is", "the", "professor", "faculty",
                      "instructor", "taking", "are", "offered", "in", "by",
                      "scds", "sol", "sob", "soai", "sas", "sot",
                      "school", "university", "saiu"}
        faculty_name = None
        for word in question.split():
            clean = word.strip("?.,")
            if len(clean) > 3 and clean.lower() not in skip_words:
                faculty_name = clean
                break

        if faculty_name:
            all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
                faculty_name, k=10, filter={"source_type": "course_list_row"}
            ))
            all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
                f"{faculty_name} course faculty teaches semester",
                k=10, filter={"source_type": "course_list_row"}
            ))

        # General catalog search — safety net
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            catalog_query, k=10, filter={"doc_category": "semester_catalog"}
        ))

    elif intent == "grading":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"doc_category": "grading"}
        ))

    elif intent == "structure" and target_school_file:
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question + " total credits degree requirements", k=15,
            filter={"filename": target_school_file}
        ))

    elif intent == "scholarship":
        scholarship_query = question + " CGPA minimum maintain scholarship criteria percentage"
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            scholarship_query, k=10, filter={"doc_category": "scholarship"}
        ))

    elif intent == "policy":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"doc_category": "policy"}
        ))

    elif intent == "calendar":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"doc_category": "calendar"}
        ))

    # B. School-Specific Search (Layer 2) — skip for semester to avoid redundancy
    if target_school_file and intent != "semester":
        all_candidates.extend(retriever.vectorstore.similarity_search_with_score(
            question, k=10, filter={"filename": target_school_file}
        ))

    # C. Global Fallback (Layer 3) — skip for school-specific semester queries
    if not (intent == "semester" and target_school_file):
        all_candidates.extend(retriever.get_relevant_chunks(question, k=10))

    # D. DE-DUPLICATION
    unique_results = []
    seen_contents = set()
    for doc, score in all_candidates:
        if doc.page_content not in seen_contents:
            unique_results.append((doc, score))
            seen_contents.add(doc.page_content)

    # E. Sort by score with catalog boost for semester intent
    if intent == "semester":
        boosted = []
        for doc, score in unique_results:
            if doc.metadata.get("doc_category") == "semester_catalog":
                boosted.append((doc, score * 0.75))
            else:
                boosted.append((doc, score))
        unique_results = sorted(boosted, key=lambda x: x[1])
    else:
        unique_results.sort(key=lambda x: x[1])

    return unique_results


# ────────────────────────────────────────────────────────────
# QUERY DECOMPOSITION
# ────────────────────────────────────────────────────────────
def decompose_query(question: str) -> list[str]:
    conjunctions = [
        " and does it ",
        " and will it ",
        " and how does ",
        " and how do ",
        " and how many ",
        " and what ",
        " and will ",
        " and can i ",
        " and is it ",
    ]
    for conj in conjunctions:
        if conj in question.lower():
            parts = question.lower().split(conj)
            if len(parts[0].split()) >= 3 and len(parts[1].split()) >= 3:
                return [parts[0].strip(), parts[1].strip()]
    return [question]


# ────────────────────────────────────────────────────────────
# CHAT ENDPOINT
# ────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Decompose compound questions
        sub_queries = decompose_query(request.question)
        raw_results = []
        for sub_q in sub_queries:
            intent = detect_intent(sub_q)
            raw_results.extend(retrieve_by_intent(intent, sub_q))

        # 2. Determine intent and school — single calculation used throughout
        primary_intent = detect_intent(request.question)
        target_school = detect_school(request.question)

        # 3. Force faculty name results via direct text scan (bypasses embedding)
        if primary_intent == "semester" and not target_school:
            skip_words = {"what", "which", "courses", "does", "teach", "this",
                          "semester", "who", "is", "the", "professor", "faculty",
                          "instructor", "taking", "are", "offered", "in", "by",
                          "scds", "sol", "sob", "soai", "sas", "sot",
                          "school", "university", "saiu"}
            for word in request.question.split():
                clean = word.strip("?.,")
                if len(clean) > 3 and clean.lower() not in skip_words:
                    forced = search_faculty_by_name(clean)
                    if forced:
                        raw_results = forced + raw_results
                    break

        # 4. Score Filtering
        score_threshold = 1.4 if primary_intent in ("semester", "scholarship") else 1.2
        filtered_results = [
            (doc, score) for doc, score in raw_results if score < score_threshold
        ]

        # 4b. School tag filtering — remove other-school chunks for school-specific queries
        if primary_intent == "semester" and target_school:
            target_tag = target_school.replace(".pdf", "")
            school_filtered = [
                (doc, score) for doc, score in filtered_results
                if doc.metadata.get("school_tag") == target_tag
                or doc.metadata.get("source_type") == "catalog_policy"
                or (doc.metadata.get("school_tag") is None
                    and doc.metadata.get("source_type") != "course_list_row")
            ]
            if len(school_filtered) >= 3:
                filtered_results = school_filtered

        # 5. De-duplicate snippets
        unique_chunks = []
        seen_text = set()
        for doc, score in filtered_results:
            snippet = doc.page_content[:150].strip()
            if snippet not in seen_text:
                unique_chunks.append((doc, score))
                seen_text.add(snippet)

        final_selection = unique_chunks[:8]

        # 6. Handle no context found
        if not final_selection:
            return ChatResponse(
                answer="I'm sorry, I don't have that specific information in my current database. Please contact the Registrar's Office at registrar@saiuniversity.edu.in for official clarification.",
                sources=[]
            )

        # 7. Format context with smart truncation based on source type
        context = ""
        sources = []
        for doc, _ in final_selection:
            src = doc.metadata.get("filename", "Unknown")
            cat = doc.metadata.get("doc_category", "General")
            source_type = doc.metadata.get("source_type", "")

            if source_type == "course_list_row":
                truncated = doc.page_content
            elif cat == "semester_catalog":
                truncated = doc.page_content[:800]
            elif cat in ("grading", "scholarship"):
                content = doc.page_content
                for phrase in ["O SAI\nUNIVERSITY\n", "re SAI\nUNIVERSITY\n", "SAI\nUNIVERSITY\n"]:
                    content = content.replace(phrase, "")
                truncated = content.strip()[:1000]
            else:
                truncated = doc.page_content[:400]

            context += f"\n[Source: {src} | Dept: {cat}]\n{truncated}\n"
            if src not in sources:
                sources.append(src)

        # 8. Build messages with chat history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in request.chat_history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"})

        # 9. Generate with Groq — fallback to second API key on rate limit
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                fallback_client = Groq(api_key=FALLBACK_KEY)
                response = fallback_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024
                )
            else:
                raise e

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