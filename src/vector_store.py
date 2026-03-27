import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document  # ✅ Fixed: LangChain Document, not DocxDocument
from src.data_loader import load_pdf_with_ocr, classify_document
from docx import Document as DocxDocument  # ✅ python-docx Document, only for reading .docx tables

# ── Setup Embeddings ───────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)

# ── School Keywords for Metadata Tagging ──────────────────
SCHOOL_KEYWORDS = {
    "business": "school_business",
    "law": "school_law",
    "computing": "school_computing_datascience",
    "data science": "school_computing_datascience",
    "media": "school_media",
    "technology": "school-technology",
    "arts": "school-arts_sciences",
    "artificial intelligence": "school_artificial_intelligence"
}

# ── Course code prefix → school tag mapping ──
COURSE_CODE_SCHOOL_MAP = {
    "CDS": "school_computing_datascience",
    "CS": "school_computing_datascience",
    "SL": "school_law",
    "BBA": "school_business",
    "BCM": "school_business",
    "SAI": "school-arts_sciences",
    "SOT": "school-technology",
    "MED": "school_media",
    "AIM": "school_artificial_intelligence",
}

# ── Catalog Course List Extractor ─────────────────────────
def extract_catalog_course_list(file_path: str) -> list:
    """
    Extracts course list table rows directly from the catalog docx.
    These contain: course title, code, credits, semester, faculty.
    Produces one LangChain Document per row for precise retrieval.
    """
    doc = DocxDocument(file_path)
    chunks = []

    for table in doc.tables:
        headers = [cell.text.strip().lower() for cell in table.rows[0].cells]

        # Identify course list tables by their headers
        if "faculty" in headers or "course title" in headers:
            for row in table.rows[1:]:  # Skip header row
                cells = [cell.text.strip() for cell in row.cells]

                # Skip empty rows
                if not any(cells):
                    continue

                # Build a natural language chunk from the row
                row_text = " | ".join(
                    f"{table.rows[0].cells[i].text.strip()}: {cells[i]}"
                    for i in range(len(cells))
                    if cells[i] and table.rows[0].cells[i].text.strip()
                )

                if row_text.strip():
                    # Skip junk rows — version/header artifacts from detail tables
                    junk_indicators = ["version", "co-1", "co-2", "co-3", "co-4",
                                       "po-1", "po-2", "pso-1", "outline teaching"]
                    if any(j in row_text.lower() for j in junk_indicators):
                        continue

                    # Detect school tag from course code prefix
                    school_tag = None
                    for prefix, tag in COURSE_CODE_SCHOOL_MAP.items():
                        if f"Course Code: {prefix}" in row_text:
                            school_tag = tag
                            break

                    # Also fall back to keyword matching in row text
                    if not school_tag:
                        row_lower = row_text.lower()
                        for keyword, tag in SCHOOL_KEYWORDS.items():
                            if keyword in row_lower:
                                school_tag = tag
                                break

                    metadata = {
                        "filename": "Course Catalogue_25-26_Sem2.docx",
                        "doc_category": "semester_catalog",
                        "source_type": "course_list_row"
                    }
                    if school_tag:
                        metadata["school_tag"] = school_tag

                    chunks.append(Document(
                        page_content=row_text,
                        metadata=metadata
                    ))

    print(f"   📋 Extracted {len(chunks)} course list rows from catalog")
    return chunks

def extract_catalog_policies(file_path: str) -> list:
    """
    Extracts the general instructions and policy paragraphs from the catalog.
    These contain: add/drop deadline, withdrawal policy, audit policy, min credits.
    """
    doc = DocxDocument(file_path)
    chunks = []
    
    # These are the exact paragraph headings in the catalog general instructions
    policy_sections = [
        "withdrawal and auditing",
        "withdrawal from a course",
        "auditing a course",
        "audit-to-credit",
        "credit-to-audit",
        "general instructions",
        "minimum",
        "add/drop",
        "add or drop",
        "please note"
    ]
    
    current_section = []
    current_heading = ""
    capture = False
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        text_lower = text.lower()
        
        # Check if this paragraph starts a policy section
        is_policy_heading = any(keyword in text_lower for keyword in policy_sections)
        
        if is_policy_heading:
            # Save previous section if it exists
            if current_section:
                combined = f"{current_heading}\n" + "\n".join(current_section)
                chunks.append(Document(
                    page_content=combined,
                    metadata={
                        "filename": "Course Catalogue_25-26_Sem2.docx",
                        "doc_category": "semester_catalog",
                        "source_type": "catalog_policy"
                    }
                ))
            current_heading = text
            current_section = []
            capture = True
        elif capture:
            current_section.append(text)
            # Stop capturing after 15 lines to avoid oversized chunks
            if len(current_section) >= 15:
                combined = f"{current_heading}\n" + "\n".join(current_section)
                chunks.append(Document(
                    page_content=combined,
                    metadata={
                        "filename": "Course Catalogue_25-26_Sem2.docx",
                        "doc_category": "semester_catalog",
                        "source_type": "catalog_policy"
                    }
                ))
                current_section = []
                capture = False

    # Save last section
    if current_section:
        combined = f"{current_heading}\n" + "\n".join(current_section)
        chunks.append(Document(
            page_content=combined,
            metadata={
                "filename": "Course Catalogue_25-26_Sem2.docx",
                "doc_category": "semester_catalog",
                "source_type": "catalog_policy"
            }
        ))

    print(f"   📋 Extracted {len(chunks)} policy chunks from catalog")
    return chunks

# ── Main Vector Store Builder ─────────────────────────────
def build_vector_store():
    data_path = "data/"
    all_chunks = []

    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    print("[INFO] Processing documents...")
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        if file.endswith((".pdf", ".docx")):
            loader = PyPDFLoader(full_path) if file.endswith(".pdf") else Docx2txtLoader(full_path)
            docs = loader.load()

            # Unified classification
            category = classify_document(file)

            # Check if PDF text is empty → use OCR
            combined_text = " ".join([d.page_content for d in docs]).strip()
            if file.endswith(".pdf") and len(combined_text) < 100:
                print(f"   🔍 Empty text detected, running OCR on {file}...")
                docs = load_pdf_with_ocr(full_path)

            # Try semantic chunking first
            chunks = text_splitter.split_documents(docs)

            # Fallback if semantic chunker produced too few chunks
            if len(chunks) <= 2:
                print(f"   ⚠️ Semantic chunking failed for {file}, using fallback splitter.")
                chunks = fallback_splitter.split_documents(docs)

            # Reset school tag for each new file
            current_school_tag = None

            # Apply metadata AFTER final chunks are decided
            for chunk in chunks:
                chunk.metadata["filename"] = file
                chunk.metadata["doc_category"] = category

                if category == "semester_catalog":
                    content_lower = chunk.page_content.lower()
                    for keyword, school_tag in SCHOOL_KEYWORDS.items():
                        if keyword in content_lower:
                            current_school_tag = school_tag
                            break
                    if current_school_tag:
                        chunk.metadata["school_tag"] = current_school_tag

            all_chunks.extend(chunks)
            print(f" ✅ {file} -> {len(chunks)} chunks")

    # ── Special handling for Course Catalogue ──
    catalog_path = os.path.join(data_path, "Course Catalogue_25-26_Sem2.docx")
    if os.path.exists(catalog_path):
        
        # 1. Extract course list rows (faculty, credits, course codes)
        course_list_chunks = extract_catalog_course_list(catalog_path)
        for chunk in course_list_chunks:
            content_lower = chunk.page_content.lower()
            for keyword, school_tag in SCHOOL_KEYWORDS.items():
                if keyword in content_lower:
                    chunk.metadata["school_tag"] = school_tag
                    break
        all_chunks.extend(course_list_chunks)
        print(f" ✅ Added {len(course_list_chunks)} course list row chunks from catalog")

        # 2. Extract policy paragraphs (deadlines, withdrawal, audit, min credits)
        policy_chunks = extract_catalog_policies(catalog_path)
        all_chunks.extend(policy_chunks)
        print(f" ✅ Added {len(policy_chunks)} catalog policy chunks")

    # ── Build Vector Store ────────────────────────────────
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="chroma_db/"
    )
    print(f"\n[INFO] ✅ Vector store built with {len(all_chunks)} chunks.")


if __name__ == "__main__":
    build_vector_store()