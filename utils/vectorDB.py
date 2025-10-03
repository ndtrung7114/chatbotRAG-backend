from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from google import genai
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Init client (once)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-chatbot"  # Matches your dashboard name

# Create if not exists (idempotent; only runs first time)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def create_retriever(chunks, embeddings):
    # Connect to the index
    index = pc.Index(index_name)

    # Get index stats to check for existing namespaces/vectors
    stats = index.describe_index_stats()

    # If there are any namespaces (indicating vectors exist somewhere), delete all in the default namespace
    if 'namespaces' in stats and len(stats['namespaces']) > 0:
        index.delete(delete_all=True, namespace="")

    vector_store = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name, namespace=""
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def load_retriever(embeddings):
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings, namespace="")
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})