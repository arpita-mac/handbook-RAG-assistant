from sentence_transformers import SentenceTransformer

# this downloads a small embedding model the first time (~80MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Employees get 18 paid leaves per year",
    "Staff are entitled to 18 days off annually",
    "The office serves coffee every morning"
]

embeddings = model.encode(sentences)

print("Shape of one embedding:", embeddings[0].shape)
print("First 5 numbers of embedding 1:", embeddings[0][:5])