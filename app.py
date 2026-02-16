"""
HYBRID RAG SYSTEM (VECTOR + KNOWLEDGE GRAPH)

Features:
- Document ingestion
- FAISS vector search
- Neo4j Knowledge Graph
- Hybrid retrieval
- LLM answering

Author: Vikash Hybrid RAG
"""

import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# -----------------------------
# Global Objects
# -----------------------------
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

VECTOR_PATH = "faiss_index"


# -----------------------------
# Load & Split Docs
# -----------------------------
def load_documents():
    loader = TextLoader("data/docs.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


# -----------------------------
# Build Vector DB
# -----------------------------
def build_vector_db():
    docs = load_documents()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_PATH)
    print("✅ Vector DB created")


# -----------------------------
# Simple Entity Extraction
# -----------------------------
def extract_entities(text):
    words = re.findall(r"[A-Z][a-zA-Z]+", text)
    return list(set(words))


# -----------------------------
# Build Knowledge Graph
# -----------------------------
def build_knowledge_graph():
    docs = load_documents()

    with driver.session() as session:
        for doc in docs:
            entities = extract_entities(doc.page_content)

            for ent in entities:
                session.run(
                    "MERGE (:Entity {name:$name})",
                    name=ent
                )

    print("✅ Knowledge Graph created")


# -----------------------------
# KG Search
# -----------------------------
def search_kg(query):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($q)
            RETURN e.name LIMIT 5
            """,
            q=query
        )
        return [r["e.name"] for r in result]


# -----------------------------
# Hybrid Retrieval
# -----------------------------
def hybrid_retrieval(query):
    db = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    vector_docs = db.similarity_search(query, k=4)
    kg_results = search_kg(query)

    context = ""

    for d in vector_docs:
        context += d.page_content + "\n"

    context += "\n--- KG FACTS ---\n"
    context += "\n".join(kg_results)

    return context


# -----------------------------
# Ask LLM
# -----------------------------
def ask_llm(query):
    context = hybrid_retrieval(query)

    prompt = f"""
You are a helpful AI assistant.
Use the context below to answer.

Context:
{context}

Question: {query}
Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    if not os.path.exists(VECTOR_PATH):
        build_vector_db()
        build_knowledge_graph()

    print("\n🔥 Hybrid RAG Ready (Vector + KG)")
    print("Type 'exit' to quit\n")

    while True:
        q = input("Ask > ")

        if q.lower() == "exit":
            break

        answer = ask_llm(q)
        print("\nAnswer:\n", answer)
