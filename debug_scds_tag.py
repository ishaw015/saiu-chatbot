from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Check what school tags exist on course_list_row chunks
results = vectorstore.get(where={"source_type": "course_list_row"})
tags = set()
scds_samples = []

for doc, meta in zip(results['documents'], results['metadatas']):
    tag = meta.get('school_tag', 'NO TAG')
    tags.add(tag)
    if 'computing' in tag or 'scds' in tag.lower() or 'cds' in doc.lower():
        scds_samples.append((tag, doc[:120]))

print(f"All school tags found: {tags}\n")
print(f"SCDS-related samples ({len(scds_samples)} found):")
for tag, doc in scds_samples[:10]:
    print(f"  Tag: {tag}")
    print(f"  Content: {doc}")
    print()