from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from typing import List, Any
import re


from langchain_community.document_loaders import (
    PyMuPDFLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document


# ─────────────────────────────────────────────
# Document Classification
# ─────────────────────────────────────────────
def classify_document(filename: str) -> str:
    name = filename.lower()

    if "catalog" in name or "catalogue" in name or "semester" in name:
        return "semester_catalog"
    elif "school_" in name or "school-" in name:
        return "school_overview"          # ← already existed, covers all school files
    elif "policy" in name or "it_policy" in name or "library" in name or "code_of_conduct" in name:
        return "policy"
    elif "calendar" in name:
        return "calendar"
    elif "scholarship" in name:
        return "scholarship"
    elif "grade" in name or "cgpa" in name:
        return "grading"
    elif "timetable" in name:
        return "timetable"
    elif "handbook" in name or "hand_book" in name:
        return "student_handbook"
    elif "holiday" in name or "holidays" in name:
        return "calendar"
    else:
        return "general"

# ─────────────────────────────────────────────
# OCR Loader
# ─────────────────────────────────────────────
def load_pdf_with_ocr(pdf_path: str) -> List[Any]:
    print(f"   🔍 Running OCR on {Path(pdf_path).name}...")
    images = convert_from_path(pdf_path)

    docs = []
    for i, image in enumerate(images):
        raw_text = pytesseract.image_to_string(
            image,
            config="--oem 3 --psm 6"
        )

        cleaned_text = re.sub(r"\n{2,}", "\n", raw_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = cleaned_text.strip()

        docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_path,
                    "filename": Path(pdf_path).name,
                    "document_type": "pdf_ocr",
                    "doc_category": classify_document(Path(pdf_path).name),
                    "page": i + 1,
                }
            )
        )

    return docs


# ─────────────────────────────────────────────
# Main Loader
# ─────────────────────────────────────────────
def load_all_documents(data_dir: str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    print(f"[INFO] Loading documents from: {data_path}\n")

    documents = []

    # ───────── PDF FILES ─────────
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF file(s):")

    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            pages = loader.load()

            # Check if text is empty → fallback to OCR
            combined_text = " ".join([p.page_content for p in pages]).strip()

            if len(combined_text) < 100:
                ocr_docs = load_pdf_with_ocr(str(pdf_file))
                documents.extend(ocr_docs)
                print(f"   ✅ {pdf_file.name} — loaded with OCR")
            else:
                doc_category = classify_document(pdf_file.name)

                for page in pages:
                    page.metadata["filename"] = pdf_file.name
                    page.metadata["document_type"] = "pdf"
                    page.metadata["doc_category"] = doc_category

                documents.extend(pages)
                print(f"   ✅ {pdf_file.name} — loaded page-wise ({len(pages)} pages)")

        except Exception as e:
            print(f"   ❌ Failed to load {pdf_file.name}: {e}")

    # ───────── DOCX FILES ─────────
    docx_files = list(data_path.glob("**/*.docx"))
    print(f"\n[INFO] Found {len(docx_files)} Word document(s):")

    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()

            for doc in loaded:
                doc.metadata["filename"] = docx_file.name
                doc.metadata["document_type"] = "docx"
                doc.metadata["doc_category"] = classify_document(docx_file.name)

            documents.extend(loaded)
            print(f"   ✅ {docx_file.name} — {len(loaded)} document(s) loaded")

        except Exception as e:
            print(f"   ❌ Failed to load {docx_file.name}: {e}")
    
    print(f"\n[INFO] Successfully loaded {len(documents)} total document(s).")
    return documents  

# ─────────────────────────────────────────────
# Test Block
# ─────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_all_documents("data")

    print("\n── Sample Document ──")
    if docs:
        print(docs[0].metadata)
        print(docs[0].page_content[:500])