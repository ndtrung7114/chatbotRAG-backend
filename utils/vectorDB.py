from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings
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
        dimension=384,  # MiniLM dims
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def create_retriever(chunks, embeddings):
    pc.Index(index_name).delete(delete_all=True)
    vector_store = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def load_retriever(embeddings):
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})