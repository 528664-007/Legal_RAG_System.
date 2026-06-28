markdown
# ⚖️ Legal RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2.11-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.3-orange.svg)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Advanced Retrieval-Augmented Generation for Legal Document Analysis**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Architecture](#-architecture) • [Queries](#-test-queries)

</div>

---

## 📖 Overview

The **Legal RAG System** is a production-ready Retrieval-Augmented Generation pipeline specifically engineered for legal document analysis. It combines **hierarchical chunking**, **hybrid search** (vector + BM25), and **legal-optimized prompting** to deliver accurate, cited answers to complex legal questions.

Unlike generic RAG systems that lose context with standard chunking, our system preserves the hierarchical structure of legal documents through a parent-child retrieval approach, ensuring that precise matches on specific clauses always return the complete legal context.

### 🎯 Problem Solved

Legal documents have unique challenges that standard RAG systems fail to address:
- **Hierarchical structure**: Articles → Sections → Subsections
- **Precise terminology**: Exact legal definitions matter
- **Cross-referencing**: Clauses reference other clauses
- **Context dependency**: A single sentence often requires the full section to understand

Our system solves these by retrieving small chunks for precise matching while returning large parent contexts to the LLM, ensuring no legal nuance is lost.

---

## ✨ Features

### 🏗️ **Hierarchical Parent-Child Chunking**
- **Parent Chunks** (1200 tokens): Preserve complete legal sections with full context
- **Child Chunks** (250 tokens): Enable precise vector similarity matching
- **Automatic Linking**: Matched children map to parent contexts for LLM processing

### 🔍 **Hybrid Search Architecture**
- **Vector Search (70%)**: Semantic understanding via `all-MiniLM-L6-v2` embeddings
- **BM25 Search (30%)**: Exact legal term matching for citations and defined terms
- **Ensemble Retrieval**: Weighted combination with configurable ratios

### ⚖️ **Legal-Optimized Prompting**
- **Strict Grounding**: Answers ONLY from retrieved context
- **Mandatory Citations**: Section/Article references with direct quotes
- **Automatic Disclaimer**: Standard legal disclaimer on every response
- **Hierarchical Awareness**: Recognizes legal document structure

### 🤖 **Multi-LLM Support**
| Provider | Model | Type | Speed |
|----------|-------|------|-------|
| **Groq** | `llama-3.1-8b-instant` | Cloud | ⚡ Fast |
| **Groq** | `llama-3.3-70b-versatile` | Cloud | 🚀 Powerful |
| **Ollama** | `llama3:8b-instruct` | Local | 🏠 Private |
| **OpenAI** | `gpt-3.5-turbo` | Cloud | 🌐 General |

### 📊 **Additional Features**
- Interactive CLI with rich formatting
- Query logging to JSON for analysis
- Configurable via `.env` file
- Comprehensive test suite
- Modular architecture for easy extension

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip
- Git

### Step 1: Clone the Repository
bash
git clone https://github.com/528664-007/Legal-RAG-System.git
cd legal-rag-system


### Step 2: Create Virtual Environment
bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate


### Step 3: Install Dependencies
bash
pip install -r requirements.txt


### Step 4: Configure Environment
bash
cp .env.example .env
# Edit .env with your settings


### Step 5: Set Up LLM Provider

<details>
<summary><b>Option A: Groq (Recommended - Free Tier)</b></summary>

bash
# 1. Get a free API key at https://console.groq.com
# 2. Add to .env:
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant

# 3. Install Groq package
pip install groq

</details>

<details>
<summary><b>Option B: Ollama (Local & Private)</b></summary>

bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start server and pull model
ollama serve
ollama pull llama3:8b-instruct

# 3. Set in .env:
LLM_PROVIDER=ollama

</details>

<details>
<summary><b>Option C: OpenAI</b></summary>

bash
# Add to .env:
OPENAI_API_KEY=sk-your_key_here
LLM_PROVIDER=openai

</details>

---

## 🎯 Quick Start

### Run the Demo
bash
python main.py --demo

This ingests 3 sample legal documents and runs 7 test queries.

### Interactive Mode
bash
python main.py

Start an interactive session with pre-loaded sample documents.

### Single Query
bash
python main.py --query "What are the remedies for breach of confidentiality?"


### Custom Document
bash
python main.py --ingest my_contract.txt --title "Service Agreement"


---

## 📚 Usage

### Command Line Options

| Flag | Description | Example |
|------|-------------|---------|
| `--demo` | Run demonstration with test queries | `python main.py --demo` |
| `--query "text"` | Process a single query | `python main.py --query "What is a trade secret?"` |
| `--ingest file.txt` | Ingest a custom document | `python main.py --ingest contract.txt` |
| `--title "Name"` | Set document title | `python main.py --ingest file.txt --title "NDA"` |
| `--provider` | Choose LLM provider | `python main.py --provider groq` |
| `--model` | Specify LLM model | `python main.py --model llama-3.3-70b-versatile` |
| `--interactive` | Start interactive mode | `python main.py --interactive` |
| `--reset` | Reset vector store | `python main.py --ingest file.txt --reset` |

### Interactive Commands

Legal Query > help          # Show help
Legal Query > stats         # Show system statistics
Legal Query > config        # Show current configuration
Legal Query > clear         # Clear the screen
Legal Query > exit          # Exit


### Python API
python
from legal_rag import LegalRAGPipeline

# Initialize
pipeline = LegalRAGPipeline()

# Ingest documents
pipeline.ingest_document(
    text="Your legal document text here...",
    title="Sample Contract"
)

# Query
result = pipeline.query("What are the termination conditions?")
print(result["response"])


---

## 🏗️ Architecture


┌─────────────────────────────────────────────────────────────────┐
│                     LEGAL RAG SYSTEM ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Document   │    │   Document   │    │   Document   │       │
│  │   Ingestor   │    │   Processor  │    │   Chunker    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Parent-Child Document Store                 │    │
│  │  ┌──────────────────┐    ┌──────────────────────────┐   │    │
│  │  │  Parent Chunks   │◄───│     Child Chunks          │   │    │
│  │  │  (1200 tokens)   │    │     (250 tokens)          │   │    │
│  │  │  Full Sections   │    │  Precise Search Units     │   │    │
│  │  └──────────────────┘    └──────────┬───────────────┘   │    │
│  └─────────────────────────────────────┼───────────────────┘    │
│                                        │                        │
│         ┌──────────────────────────────┼──────────────────┐    │
│         │         Hybrid Retrieval     │                  │    │
│         │  ┌────────────┐  ┌──────────┴───────────┐      │    │
│         │  │  Vector    │  │       BM25          │      │    │
│         │  │  Search    │  │    Keyword Search    │      │    │
│         │  │  (70%)     │  │       (30%)         │      │    │
│         │  └────────────┘  └──────────┬───────────┘      │    │
│         └─────────────────────────────┼──────────────────┘    │
│                                       │                        │
│                                       ▼                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Ensemble Retriever                     │    │
│  │              Child Match → Parent Context                │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Legal Prompt Template                       │    │
│  │  • Strict Grounding  • Citations  • Disclaimer          │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   LLM (Groq/Ollama/OpenAI)               │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Cited Legal Response                        │    │
│  │  Answer + Section Citations + Disclaimer                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘


### Data Flow
1. **Document Ingestion** → Split into large parent chunks (preserve context)
2. **Child Chunking** → Further split parents into small searchable units
3. **Embedding** → Generate 384-dimensional vectors via `all-MiniLM-L6-v2`
4. **Indexing** → Store in ChromaDB with metadata (section, article, clause type)
5. **Query Processing** → Hybrid search (70% vector + 30% BM25) on child chunks
6. **Parent Mapping** → Map matched children to parent documents
7. **Context Assembly** → Format parent contexts with legal headers
8. **LLM Generation** → Generate cited response with legal disclaimer

---

## 📁 Project Structure


legal-rag-system/
│
├── legal_rag/                      # Core package
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Configuration management
│   ├── pipeline.py                 # Main RAG pipeline orchestrator
│   ├── document_processor.py       # Legal document chunking
│   ├── retrievers.py               # BM25 & hybrid retrievers
│   ├── vector_store.py             # ChromaDB management
│   ├── llm_provider.py             # LLM abstraction (Groq/Ollama/OpenAI)
│   ├── embeddings.py               # Sentence transformer embeddings
│   ├── prompts.py                  # Legal-optimized prompt templates
│   └── utils.py                    # Utility functions & formatting
│
├── data/                           # Sample documents
│   ├── __init__.py
│   └── sample_contracts.py         # NDA, Statute, Employment Agreement
│
├── scripts/                        # Utility scripts
│   ├── __init__.py
│   └── run_demo.py                 # Demonstration script
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_document_processor.py  # Chunking tests
│   └── test_retrievers.py          # Retrieval tests
│
├── chroma_db/                      # Vector store (auto-created)
│   └── .gitkeep
│
├── logs/                           # Query logs
│   └── .gitkeep
│
├── main.py                         # CLI entry point
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
├── .gitignore
└── README.md


---

## ⚙️ Configuration

All settings are configured via `.env` file:

| Category | Setting | Default | Description |
|----------|---------|---------|-------------|
| **LLM** | `LLM_PROVIDER` | `groq` | LLM backend |
| | `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| | `GROQ_API_KEY` | - | Groq API key |
| | `OLLAMA_MODEL` | `llama3:8b-instruct` | Ollama model |
| | `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model |
| **Embeddings** | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| | `EMBEDDING_DEVICE` | `cpu` | Device (cpu/cuda) |
| **Chunking** | `CHILD_CHUNK_SIZE` | `250` | Child chunk size |
| | `PARENT_CHUNK_SIZE` | `1200` | Parent chunk size |
| **Retrieval** | `VECTOR_WEIGHT` | `0.7` | Vector search weight |
| | `BM25_WEIGHT` | `0.3` | BM25 weight |
| | `TOP_K` | `5` | Child chunks to retrieve |
| | `PARENT_TOP_K` | `3` | Parent contexts to return |
| **Storage** | `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store location |

---

## 🧪 Test Queries

Here are categorized queries to test the system's capabilities:

### 📋 NDA & Confidentiality

python
# Core Confidentiality
"What are the obligations of the receiving party regarding confidential information?"
"What constitutes confidential information under the NDA?"
"Can the receiving party share confidential information with employees?"

# Breach & Remedies
"What remedies are available if the receiving party breaches confidentiality?"
"Can the disclosing party get an injunction without proving actual damages?"
"What are the legal consequences of unauthorized disclosure of confidential information?"

# Exceptions
"What are the exceptions to confidentiality obligations under the NDA?"
"When is confidential information no longer protected?"
"What if the information was already publicly available?"

# Duration & Termination
"How long does the confidentiality obligation survive after termination of the agreement?"
"What happens to confidential information after the NDA expires?"
"How long does the receiving party have to return confidential materials?"

# Governing Law
"Which state law governs the confidentiality agreement?"
"Where must disputes under the NDA be resolved?"


### ⚖️ Trade Secrets (California Civil Code)

python
# Definitions
"What constitutes a trade secret under California law?"
"What is the legal definition of misappropriation of trade secrets?"
"What are considered improper means of acquiring trade secrets?"

# Injunctive Relief
"Can a court issue an injunction for trade secret misappropriation?"
"How long can an injunction last for trade secret protection?"
"Can an injunction continue after the trade secret ceases to exist?"

# Damages
"What damages can be recovered for trade secret misappropriation?"
"How are damages calculated for trade secret theft?"
"Can you get exemplary damages for willful and malicious misappropriation?"

# Attorney's Fees
"When can attorney's fees be awarded in trade secret cases?"
"What happens if a trade secret claim is made in bad faith?"


### 💼 Employment Agreement

python
# Employment Terms
"What are the duties of the employee under the employment agreement?"
"How long does the employment agreement last?"
"Is the employee allowed to work other jobs during employment?"

# Compensation
"What is the employee's base salary under the employment agreement?"
"How much annual bonus can the employee earn?"
"What benefits does the employee receive?"
"How many stock options is the employee granted and when do they vest?"

# Intellectual Property
"What happens to intellectual property created during employment?"
"Who owns inventions created by the employee?"
"Does the employee have to assign intellectual property rights to the company?"

# Confidentiality (Employment)
"How long does employee confidentiality last after termination?"
"What confidential information must the employee protect?"
"What must the employee return upon termination of employment?"

# Termination
"What are the grounds for termination for cause under the employment agreement?"
"Can the employee be terminated without cause?"
"What is the notice period for termination?"
"What is termination for good reason?"
"What happens to salary and benefits upon termination?"

# Restrictive Covenants
"What are the non-compete restrictions in the employment agreement?"
"How long does the non-compete last after termination?"
"Can the employee solicit other employees after leaving the company?"
"Are there non-solicitation restrictions on customers?"
"What is the geographic scope of the non-compete clause?"

# Dispute Resolution
"How are employment disputes resolved under the agreement?"
"Is arbitration required for employment disputes?"
"What law governs the employment agreement?"


### 🔍 Cross-Document Comparisons

python
# NDA vs Statute
"How does the NDA definition of confidential information compare to trade secret law?"
"Are the remedies in the NDA broader than California statutory remedies?"
"How do confidentiality exceptions differ between the NDA and the statute?"

# NDA vs Employment Agreement
"How do confidentiality obligations differ between the NDA and employment agreement?"
"Which agreement has longer confidentiality survival periods?"
"Are the remedies different between the NDA and employment agreement?"

# All Documents
"What protections exist across all documents for confidential information?"
"How do the dispute resolution provisions differ across the documents?"
"Which document provides the strongest remedies for breach?"


### 🎯 Practical Scenarios

python
"If an employee accidentally emails confidential information to a competitor, what remedies apply?"
"Can a former employee use trade secrets at a new job?"
"What happens if someone reverse engineers a trade secret?"
"If confidential information becomes public through no fault of the receiving party, what happens?"
"Can a company prevent a former employee from working for a competitor?"
"What are the maximum potential damages for breaching the NDA?"
"What steps must be taken to maintain trade secret protection?"


### 🧪 Edge Cases

python
"What if the breach was unintentional?"
"Can you partially breach confidentiality?"
"What if both parties breach at the same time?"
"What if state law conflicts with the NDA provisions?"
"What if the company goes bankrupt during the NDA term?"
"Can confidentiality obligations be inherited by a successor company?"


---

## 🧪 Running Tests

bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_document_processor.py -v

# Run with coverage
pytest --cov=legal_rag tests/


---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Embedding Dimension | 384 (all-MiniLM-L6-v2) |
| Child Chunk Size | 250 characters |
| Parent Chunk Size | 1200 characters |
| Vector Search Weight | 70% |
| BM25 Search Weight | 30% |
| Top-K Children | 5 |
| Top-K Parents | 3 |
| LLM Temperature | 0.1 |
| Max Response Tokens | 1024 |

---

## 🔧 Troubleshooting

<details>
<summary><b>NLTK Data Error</b></summary>

bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

</details>

<details>
<summary><b>Groq Model Decommissioned</b></summary>

Update `.env` to use a current model:
bash
GROQ_MODEL=llama-3.1-8b-instant

Current models: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`
</details>

<details>
<summary><b>ChromaDB Import Warning</b></summary>

In `legal_rag/vector_store.py`, change:
python
from langchain_community.vectorstores import Chroma

</details>

<details>
<summary><b>Embedding Model Download Stuck</b></summary>

The first run downloads `all-MiniLM-L6-v2` (~80MB). This is cached for subsequent runs.
</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This system is for **informational purposes only** and does **not constitute legal advice**. Legal outcomes depend on specific jurisdictional requirements and individual circumstances. Always consult with a qualified legal professional for advice regarding your specific situation.

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ star on GitHub!

---

<div align="center">

**Built with ❤️ using LangChain, ChromaDB, and Groq**

[Report Bug](https://github.com/yourusername/legal-rag-system/issues) • [Request Feature](https://github.com/yourusername/legal-rag-system/issues)

</div>


This README provides:
- Clear project overview and problem statement
- Detailed installation instructions for all LLM providers
- Architecture diagram showing data flow
- Comprehensive test queries organized by category
- Configuration reference table
- Troubleshooting guide
- Professional badges and formatting

Replace `yourusername` with your actual GitHub username before publishing!
