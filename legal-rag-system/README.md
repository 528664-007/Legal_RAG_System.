# ⚖️ Legal RAG System (Groq-Powered)

Advanced Retrieval-Augmented Generation for legal document analysis,
using **Groq** as the LLM backend for fast cloud inference.

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
cp .env.example .env
# Edit .env → GROQ_API_KEY=your_key_here

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. Run
python main.py                # interactive mode
python main.py --demo         # full demo
python main.py --query "What are the remedies for breach of confidentiality?"
```

## Key Features
- **Parent-Child Chunking** — precise vector matching + full legal context
- **Hybrid Search** — 70% vector similarity + 30% BM25 keyword matching
- **Groq LLM** — fast `llama3-8b-8192` inference
- **Legal Prompting** — mandatory citations, disclaimers, strict grounding
- **Interactive CLI** — query loop with stats and logging

## Configuration (.env)
| Key | Default | Description |
|-----|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Get at console.groq.com |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer |
| `TOP_K` | `5` | Child chunks retrieved |
| `PARENT_TOP_K` | `3` | Parent contexts returned |

## CLI Usage
```
python main.py --demo                         # 7 test queries on 3 sample docs
python main.py --query "What is a trade secret?"
python main.py --ingest contract.txt --title "My Contract"
python main.py --provider groq --interactive
```

## Testing
```bash
pytest tests/ -v
```

## Disclaimer
For informational purposes only. Does not constitute legal advice.
