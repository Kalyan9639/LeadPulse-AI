# AI Lead Generation & Intelligence Extraction Pipeline

A modular, plug-and-play AI system that automatically classifies customer-sales conversation transcripts, indexes and retrieves key dialogue context via local embeddings and vector search, strictly extracts structured lead intelligence with zero hallucination using Groq, and persists the results into a styled Excel workbook.

---

## System Architecture

```
                               ┌─────────────────────────────┐
                               │ Conversation Transcript     │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   ML Gatekeeper Classifier  │
                               │  (HuggingFace: mr-checker/  │
                               │  SalesGen balanced model)   │
                               └──────────────┬──────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │                                             │
               Prediction = 0                                 Prediction = 1
                       │                                             │
                       ▼                                             ▼
        ┌─────────────────────────────┐               ┌─────────────────────────────┐
        │  Log: Not a Potential Lead  │               │   Transcript Segmentation   │
        │      (Skip Extraction)      │               │       & Turn Chunking       │
        └─────────────────────────────┘               └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │  Local Embedding Generation │
                                                      │  (Ollama: granite-embedding)│
                                                      └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │  Vector Index & ChromaDB    │
                                                      │  Static 3-Query Retrieval   │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │   Groq LLM (gpt-oss-20b)    │
                                                      │    Native JSON Extraction   │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │    Excel Lead Storage       │
                                                      │      (output/leads.xlsx)    │
                                                      └─────────────────────────────┘
```

---

## Target Features Extracted

The extractor adheres to strict zero-hallucination constraints: if any feature is not explicitly mentioned, it is left empty (`""`).

| Feature | Description | Example |
| :--- | :--- | :--- |
| **`Name`** | Customer's full or first name | `Rohan Mehta` |
| **`Phone no.`** | Customer's phone number (if given) | `""` *(empty if not provided)* |
| **`Source`** | Marketing / lead acquisition channel | `Instagram` |
| **`Last Discussed topic`** | Factual summary of the final discussion point | `Scheduling demo and sending case studies` |
| **`Status`** | Lead qualification status | `Follow-up Pending` |
| **`Assigned to`** | Sales representative name | `Priya` |
| **`Follow up?`** | Agreed next steps and follow-up timeline | `Follow up on WhatsApp by tomorrow evening` |

---

## Project Structure & Modularity (Plug & Play)

Each component is encapsulated in its own standalone module with clean abstractions:

```
├── config.py                 # Centralized configurations and environment variables
├── classifier.py             # LeadClassifier: scikit-learn ML model prediction
├── rag_pipeline.py           # TranscriptRAG: Ollama embeddings + ChromaDB vector retrieval
├── extractor.py              # LeadExtractor: Groq openai/gpt-oss-20b structured JSON extraction
├── excel_manager.py          # ExcelManager: Thread-safe openpyxl lead storage and styling
├── pipeline.py               # LeadGenPipeline: End-to-end orchestrator with concurrent execution
├── main.py                   # CLI entry point for single files and batch directories
├── requirements.txt          # Python dependencies
├── Lead_Gen_ML_Model/        # Serialized models & datasets
│   ├── sales_model_balanced.joblib
│   ├── sales_model_fp.joblib
│   └── sales_conversation_100k.csv
├── transcripts/              # Sample transcript files
│   ├── transcript.txt        # Lead sample (Positive: 1)
│   └── sample_non_lead.txt   # Non-lead sample (Negative: 0)
└── output/                   # Generated artifacts
    └── leads.xlsx            # Formatted Excel spreadsheet
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Ensure your Groq API key is present in `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Ensure Ollama is Running
Pull the embedding model:
```bash
ollama pull granite-embedding:30m
```

---

## Usage

### Run on Default Transcript
```bash
python main.py
```

### Run on a Custom Transcript File
```bash
python main.py --file ./transcripts/my_transcript.txt
```

### Run Concurrently on a Directory of Transcripts
```bash
python main.py --dir ./transcripts --workers 4 --output ./output/leads.xlsx
```

---

## Plug-and-Play Extensibility

Team members can easily replace or customize any component:

```python
from pipeline import LeadGenPipeline
from classifier import LeadClassifier
from rag_pipeline import TranscriptRAG
from extractor import LeadExtractor
from excel_manager import ExcelManager

# Custom components can be plugged in directly:
pipeline = LeadGenPipeline(
    classifier=LeadClassifier(model_path="path/to/custom_model.joblib"),
    rag_engine=TranscriptRAG(embedding_model="granite-embedding:30m"),
    extractor=LeadExtractor(model="openai/gpt-oss-20b"),
    excel_manager=ExcelManager(file_path="output/custom_leads.xlsx")
)

# Process a single conversation
result = pipeline.process_transcript("CUSTOMER: ... SALES_REP: ...")

# Or process a file
result = pipeline.process_file("transcripts/transcript.txt")
```
