!pip install sentence-transformers faiss-cpu groq pypdf
import os
os.environ["GROQ_API_KEY"] = ""
from pypdf import PdfReader

pdf_path = "5InchHIEGHT_Gain.pdf"   # your uploaded file
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print("Total characters:", len(text))
print(text[:800])  # preview
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(text)
print("Number of chunks:", len(chunks))
print(chunks[0][:400])
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("FAISS index size:", index.ntotal)
def retrieve(query, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask_pdf(query, k=3, model_name="llama3-8b-8192"):
    context = "\n\n".join(retrieve(query, k))
    prompt = f"Use the following CONTEXT to answer the QUESTION.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer clearly."

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
    print(ask_pdf("how is sleep crutial for growth?", model_name="llama-3.3-70b-versatile"))
print(ask_pdf("how are macronutrients related?", model_name="llama-3.3-70b-versatile"))
