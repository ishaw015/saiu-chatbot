from src.retriever import Retriever

def audit_db():
    print("[INFO] Connecting to database for audit...")
    try:
        retriever = Retriever()
        # Get all stored data from Chroma
        all_docs = retriever.vectorstore.get()
        metadatas = all_docs.get('metadatas', [])
        
        if not metadatas:
            print("❌ The database is empty!")
            return

        # Count chunks per file
        file_counts = {}
        category_counts = {}
        
        for meta in metadatas:
            fname = meta.get('filename', 'Unknown')
            cat = meta.get('doc_category', 'Unknown')
            file_counts[fname] = file_counts.get(fname, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"\n{'='*40}")
        print(f"📊 DATABASE AUDIT: {len(metadatas)} TOTAL CHUNKS")
        print(f"{'='*40}")
        
        print("\n📂 CHUNKS PER FILE:")
        for file, count in sorted(file_counts.items()):
            print(f"  - {file}: {count} chunks")
            
        print("\n🏷️ CHUNKS PER CATEGORY:")
        for cat, count in sorted(category_counts.items()):
            print(f"  - {cat}: {count} chunks")

        # Check for the "Missing" files specifically
        missing_targets = ["grades.pdf", "SaiU_Scholarship.pdf", "Academic Calendar_25-26.docx"]
        print(f"\n{'='*40}")
        for target in missing_targets:
            if target not in file_counts:
                print(f"❗ CRITICAL: {target} is NOT in the database!")
            elif file_counts[target] < 2:
                print(f"⚠️ WARNING: {target} only has {file_counts[target]} chunk(s). Content might be unreadable.")

    except Exception as e:
        print(f"❌ Error during audit: {e}")

if __name__ == "__main__":
    audit_db()