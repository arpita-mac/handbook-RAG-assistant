import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# --- Step 1: Embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 2: ChromaDB setup ---
client = chromadb.Client()
collection = client.create_collection(name="policy_docs")

documents = [
    "Employees get 18 paid leaves per year.",
    "Sick leave requires a medical certificate if taken for more than 2 consecutive days.",
    "Leave requests must be submitted at least 3 days in advance.",
    "The office has free coffee and snacks available all day.",
    "Remote work is allowed up to 2 days per week with manager approval."
]

embeddings = model.encode(documents).tolist()

collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# --- Step 3: Retrieval ---
question = "How many days can I work from home?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

print("Question:", question)
print("\nTop matches:")
for doc in results["documents"][0]:
    print("-", doc)

# --- Step 4: LLM answer using retrieved context ---
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer using ONLY the context below. If the answer isn't in the context, say "I don't have information about that."

Context:
{context}"""),
    ("user", "{question}")
])

chain = prompt | llm

context = "\n".join(results["documents"][0])

response = chain.invoke({
    "context": context,
    "question": question
})

print("\nAnswer:", response.content)