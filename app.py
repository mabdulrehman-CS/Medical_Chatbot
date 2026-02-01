from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "medical-chatbot"  # Ensure this matches the name in store_index.py

# 1. Load Embeddings
embeddings = download_hugging_face_embeddings()

# 2. Load Existing Index
# New Method: We don't need pinecone.init(). The VectorStore handles the connection.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# 3. Setup Prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# 4. Load LLM (Updated to LlamaCpp for GGUF models)
# Make sure the file name matches exactly what you downloaded
local_model_path = "D:\\End_to_End_Medical_Chatbot\\model\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=local_model_path,
    n_ctx=2048,           # Context window size
    temperature=0.8,      # Creativity level
    max_tokens=512,       # Max answer length
    verbose=True
)

# 5. Retrieval Chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(f"User Input: {input_text}")
    
    # Updated: use .invoke() instead of calling the object directly
    result = qa.invoke({"query": input_text})
    
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)