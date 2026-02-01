# Medical Chatbot 

An intelligent AI-powered medical chatbot built using advanced LLM technology and Retrieval-Augmented Generation (RAG) to provide accurate medical information and assistance.

## 🎯 Problem Statement

Access to reliable medical information is crucial, yet many people struggle to find quick, accurate answers to their health-related questions. This project addresses the need for an accessible, intelligent medical assistant that can:
- Provide instant responses to medical queries
- Deliver information based on verified medical documentation
- Offer 24/7 availability without human intervention
- Scale to handle multiple users simultaneously

## ✨ Features

- **Intelligent Question Answering**: Uses advanced NLP to understand and respond to medical queries
- **RAG Architecture**: Combines retrieval-based and generative AI for accurate, context-aware responses
- **Vector Database Integration**: Leverages Pinecone for fast and efficient document retrieval
- **Local LLM Support**: Runs Meta Llama 3.1 8B model locally for privacy and cost-effectiveness
- **Web Interface**: Clean, user-friendly chat interface built with Flask
- **Document Processing**: Extracts and indexes information from PDF medical documents
- **Context-Aware Responses**: Maintains conversation context for better user experience
- **Source Attribution**: Returns source documents for transparency and verification

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.x**: Primary programming language
- **Flask**: Web framework for the application interface
- **LangChain**: Framework for building LLM applications
- **Meta Llama 3.1 8B (GGUF)**: Quantized large language model for efficient inference

### AI/ML Components
- **HuggingFace Transformers**: For embedding models
- **sentence-transformers/all-MiniLM-L6-v2**: Text embedding model (384 dimensions)
- **LlamaCpp**: Python bindings for running GGUF models efficiently

### Vector Database
- **Pinecone**: Cloud-based vector database for semantic search
- **Serverless Architecture**: AWS us-east-1 region with cosine similarity metric

### Document Processing
- **PyPDFLoader**: Extract text from PDF medical documents
- **RecursiveCharacterTextSplitter**: Intelligent text chunking (500 chars, 20 overlap)

### Other Tools
- **python-dotenv**: Environment variable management
- **HTML/CSS**: Frontend interface design

## 🚀 Capabilities

This medical chatbot can:

1. **Answer Medical Questions**: Provide information about diseases, symptoms, treatments, and medications
2. **Retrieve Relevant Information**: Search through medical documents to find accurate answers
3. **Handle Complex Queries**: Understand context and provide detailed, multi-faceted responses
4. **Maintain Conversation Flow**: Process follow-up questions with context awareness
5. **Scale Efficiently**: Handle multiple concurrent users.
6. **Work Offline**: Local LLM ensures privacy and reduces API costs
7. **Process Large Documents**: Efficiently chunk and index extensive medical literature
8. **Provide Source References**: Returns source documents for answer verification

## 📋 How It Works

### 1. Data Ingestion Pipeline (store_index.py)
```
PDF Documents → Text Extraction → Chunking → Embedding → Vector Storage (Pinecone)
```
- Loads medical PDF documents from the `data/` directory
- Splits text into manageable chunks (500 characters with 20 overlap)
- Generates embeddings using HuggingFace's MiniLM model
- Stores vectors in Pinecone for fast retrieval

### 2. Query Processing (app.py)
```
User Query → Embedding → Vector Search → Context Retrieval → LLM Processing → Response
```
- User submits a question through the web interface
- Query is embedded using the same model
- Pinecone finds the 2 most relevant document chunks
- Retrieved context + query are sent to Llama 3.1 8B
- LLM generates a contextual, accurate response

### 3. Prompt Engineering
- Custom prompt template ensures responses are:
  - Grounded in provided context
  - Honest about knowledge limitations
  - Focused on being helpful without hallucination

## 📁 Project Structure

```
End_to_End_Medical_Chatbot/
├── app.py                  # Main Flask application
├── store_index.py          # Vector database indexing script
├── setup.py               # Package setup configuration
├── template.py            # Project structure generator
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables (not tracked)
├── data/                 # PDF medical documents
├── model/                # Local LLM model files
│   └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
├── src/
│   ├── __init__.py
│   ├── helper.py         # Utility functions (PDF loading, embeddings)
│   └── prompt.py         # Prompt templates
├── static/
│   └── style.css         # Frontend styling
├── templates/
│   └── chat.html         # Chat interface HTML
└── research/
    └── experiment.ipynb  # Development notebooks
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Pinecone account and API key
- 8GB+ RAM (for running Llama 3.1 8B locally)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/mabdulrehman-CS/Medical_Chatbot.git
cd Medical_Chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```

5. **Download the LLM model**
- Download Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
- Place it in the `model/` directory

6. **Prepare your data**
- Add medical PDF documents to the `data/` directory

7. **Index the documents**
```bash
python store_index.py
```

8. **Run the application**
```bash
python app.py
```

9. **Access the chatbot**
- Open your browser and navigate to `http://localhost:8080`

## 🎓 Model Information

**Meta Llama 3.1 8B Instruct (Q4_K_M)**
- **Size**: ~4.58 GB quantized model
- **Context Window**: 2048 tokens
- **Quantization**: Q4_K_M (4-bit quantization for efficiency)
- **Temperature**: 0.8 (balanced creativity)
- **Max Tokens**: 512 per response

## 🔐 Security & Privacy

- Local LLM processing ensures medical queries remain private
- API keys stored in environment variables
- No data sent to third-party LLM services (except Pinecone for vector storage)

## 📊 Performance

- **Response Time**: 2-5 seconds (depends on hardware)
- **Context Retrieval**: <1 second (Pinecone)
- **Concurrent Users**: Scalable with Flask deployment options
- **Accuracy**: Grounded in source documents with minimal hallucination

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.

## ⚠️ Disclaimer

This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 👨‍💻 Author

**Muhammad Abdul Rehman**
- GitHub: [@mabdulrehman-CS](https://github.com/mabdulrehman-CS)

## 🙏 Acknowledgments

- Meta AI for Llama 3.1 model
- HuggingFace for embedding models
- Pinecone for vector database infrastructure
- LangChain community for the amazing framework