from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "medical-chatbot"  # Changed to match your app.py

# 1. Load, Split, and Embed Data
extracted_data = load_pdf("D:\\End_to_End_Medical_Chatbot\\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# 2. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 3. Create Index (if it doesn't exist)
# This prevents "Index not found" errors
existing_indexes = [index.name for index in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # Matches the MiniLM-L6-v2 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

# 4. Store Vectors
# Using from_documents is better than from_texts as it preserves metadata
print("Upserting vectors... this may take a moment.")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

print("Vectors stored successfully!")