# 🔥 Hybrid RAG System (Vector + Knowledge Graph + Web)

Hybrid RAG is an advanced **Retrieval-Augmented Generation system** that
combines:

-   Semantic Vector Search (FAISS)
-   Knowledge Graph Search (Neo4j)
-   Live Web Search (SerpAPI)

This architecture provides accurate, explainable, and wide-coverage
answers by routing queries to the best available knowledge source.

------------------------------------------------------------------------

## ✨ Key Features

-   PDF, TXT, DOCX document ingestion\
-   Automatic document chunking & embeddings\
-   FAISS Vector Database\
-   Neo4j Knowledge Graph integration\
-   Hybrid retrieval: Vector + KG\
-   Automatic Web fallback using SerpAPI\
-   Streamlit UI\
-   Grounded and low-hallucination answers

------------------------------------------------------------------------

## 🧠 How It Works

    User Question
          |
          |----> Vector Search (FAISS)
          |----> Knowledge Graph (Neo4j)
          |----> Web Search (SerpAPI)
                    |
                    ↓
            Retrieved Context (Merged)
                    |
                    ↓
                  LLM
                    |
                    ↓
               Final Answer

------------------------------------------------------------------------

## 🏗 Architecture Components

-   Document Loader -- Loads PDF, TXT, DOCX\
-   Text Splitter -- Splits documents\
-   Embedding Model -- Converts text to vectors\
-   Vector Database -- FAISS\
-   Knowledge Graph -- Neo4j\
-   Graph Builder -- Entity & relation extraction\
-   Hybrid Retriever -- Combines vector + graph\
-   Web Search -- SerpAPI\
-   LLM -- Generates answers

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   Streamlit\
-   LangChain\
-   FAISS\
-   Neo4j\
-   HuggingFace Embeddings\
-   OpenAI / Groq\
-   SerpAPI

------------------------------------------------------------------------

## 📁 Project Structure

hybrid-rag/ 
│ 
├── app.py\
├── .env\
└── README.md

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone Repository

``` bash
git clone https://github.com/vikashajithan/Hybrid_RAG.git
cd hybrid-rag
```

### 2. Create Virtual Environment

``` bash
python -m venv venv
```

Activate:

Windows

``` bash
venv\Scripts\activate
```

Mac/Linux

``` bash
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔐 Environment Variables

Create `.env` file:

    OPENAI_API_KEY=your_openai_key
    SERPAPI_API_KEY=your_serpapi_key
    NEO4J_URI=bolt+s://xxxx.neo4j.io
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password

------------------------------------------------------------------------

## ▶ Run Application

``` bash
streamlit run app.py
```

Open browser:

http://localhost:8501

------------------------------------------------------------------------

## 🧪 Example Usage

1.  Upload documents\
2.  Build vector index & KG\
3.  Ask question\
4.  Hybrid retriever searches Vector + KG + Web\
5.  Get final answer

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Graph visualization\
-   Source citations\
-   Multi-file ingestion\
-   Persistent storage\
-   Agentic Hybrid RAG

------------------------------------------------------------------------

## 📜 License

MIT License

------------------------------------------------------------------------

## 👨‍💻 Author

Vikash

Feel free to fork, star ⭐, and contribute!

