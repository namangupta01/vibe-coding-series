# RAG Application - Retrieval-Augmented Generation System

## Overview

This project implements a comprehensive Retrieval-Augmented Generation (RAG) application that ingests content from 5 research papers, creates a vector database for semantic retrieval, and powers a conversational bot with memory for contextually-aware responses.

The system is designed to answer questions about transformer-based language models using content from influential research papers including "Attention Is All You Need", BERT, GPT-3, RoBERTa, and T5.

## Features

### 🔧 Core Components
- **PDF Processing Engine**: Downloads and extracts text from research papers with intelligent chunking
- **Vector Database**: FAISS-based semantic search with sentence transformer embeddings
- **Conversational Bot**: Context-aware responses with conversation memory (last 4 interactions)
- **Evaluation Framework**: Comprehensive assessment using RAGAS and custom metrics

### 🎯 Key Capabilities
- Semantic document retrieval using state-of-the-art embeddings
- Multi-source document processing and indexing
- Interactive chat interface with conversation memory
- Comprehensive evaluation using established metrics (RAGAS)
- Automated PDF report generation
- Command-line interface for easy operation

## Architecture

```
RAG Application Architecture
├── PDF Processing Layer
│   ├── Document Download & Extraction
│   ├── Text Preprocessing & Cleaning
│   └── Intelligent Chunking
├── Vector Store Layer
│   ├── Embedding Generation (Sentence Transformers)
│   ├── FAISS Vector Database
│   └── Similarity Search
├── Conversational Layer
│   ├── Language Model Integration (DialoGPT)
│   ├── Context Retrieval & Ranking
│   └── Memory Management (4 interactions)
└── Evaluation Layer
    ├── RAGAS Metrics (Faithfulness, Relevancy, etc.)
    ├── Custom Metrics (Coherence, Context Usage)
    └── Performance Analytics
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Sufficient disk space for model downloads (~2GB)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-application
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**
   ```bash
   mkdir -p data/pdfs data/vector_store reports
   ```

4. **Run the application setup**
   ```bash
   python main.py --setup
   ```

## Usage

### Command Line Interface

The application provides a comprehensive CLI for different operations:

```bash
# Setup the application (download PDFs, create vector store)
python main.py --setup

# Start interactive chat session
python main.py --chat

# Run full evaluation with metrics
python main.py --evaluate

# Answer test questions only
python main.py --questions

# Generate PDF report (requires evaluation first)
python main.py --report

# Run complete demonstration (setup + evaluate + report)
python main.py --demo

# Display system information
python main.py --info

# Reset application and clean up
python main.py --reset
```

### Interactive Chat Example

```python
from src.rag_application import RAGApplication

# Initialize and setup
app = RAGApplication()
app.setup()  # or app.load_existing_setup()

# Start interactive chat
app.chat_interactive()
```

### Programmatic Usage

```python
from src.rag_application import RAGApplication

# Initialize application
app = RAGApplication()
app.setup()

# Single question
response = app.bot.chat("What is the Transformer architecture?")
print(response['response'])

# Run evaluation
results = app.run_evaluation()
print(f"Overall Score: {results['summary']['overall_score']}")
```

## Dataset

The application processes five influential research papers:

1. **Attention Is All You Need** (Transformer) - https://arxiv.org/pdf/1706.03762.pdf
2. **BERT: Pre-training of Deep Bidirectional Transformers** - https://arxiv.org/pdf/1810.04805.pdf
3. **Language Models are Few-Shot Learners** (GPT-3) - https://arxiv.org/pdf/2005.14165.pdf
4. **RoBERTa: A Robustly Optimized BERT Pretraining Approach** - https://arxiv.org/pdf/1907.11692.pdf
5. **Exploring the Limits of Transfer Learning with T5** - https://arxiv.org/pdf/1910.10683.pdf

## Evaluation

### Test Questions

The system is evaluated on 10 carefully crafted questions covering:
- Factual knowledge retrieval
- Conceptual understanding
- Comparative analysis
- Technical explanations
- Cross-document reasoning

### Metrics

**RAGAS Metrics:**
- **Faithfulness**: Factual consistency with source documents
- **Answer Relevancy**: Relevance of responses to questions
- **Context Precision**: Precision of retrieved context
- **Context Recall**: Recall of relevant context

**Custom Metrics:**
- **Response Length**: Optimal response length scoring
- **Context Usage**: Utilization of retrieved context
- **Coherence**: Linguistic and logical coherence
- **Response Time**: System performance and efficiency

## Configuration

Key configuration parameters in `src/config.py`:

```python
# Text processing
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Sentence transformer
LLM_MODEL = "microsoft/DialoGPT-medium"  # Language model

# Memory
MEMORY_SIZE = 4            # Conversation turns to remember
```

## Project Structure

```
rag-application/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── pdf_processor.py       # PDF download and processing
│   ├── vector_store.py        # Vector database management
│   ├── conversational_bot.py  # Chat bot with memory
│   ├── evaluator.py           # Evaluation framework
│   ├── rag_application.py     # Main application class
│   └── report_generator.py    # PDF report generation
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/                      # Data directory
    ├── pdfs/                  # Downloaded PDF files
    ├── vector_store/          # FAISS vector database
    └── reports/               # Generated reports
```

## Performance

Expected performance characteristics:
- **Setup Time**: 5-10 minutes (first run, includes downloads)
- **Query Response**: 1-3 seconds per question
- **Memory Usage**: ~2GB for models and vector store
- **Accuracy**: 70-80% based on evaluation metrics

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Ensure stable internet connection
   - Check available disk space
   - Retry setup command

2. **Memory Issues**
   - Reduce chunk size in configuration
   - Use CPU instead of GPU if memory limited
   - Close other applications

3. **PDF Download Issues**
   - Check internet connectivity
   - Verify arXiv URLs are accessible
   - Manual download to `data/pdfs/` if needed

### Logs

Application logs are available in the console output. For debugging:
```bash
python main.py --setup --verbose
```

## Extending the Application

### Adding New Documents
1. Add PDF URLs to `Config.PDF_URLS` in `src/config.py`
2. Add corresponding names to `Config.PDF_NAMES`
3. Re-run setup: `python main.py --reset && python main.py --setup`

### Custom Evaluation Metrics
Extend the `CustomMetrics` class in `src/evaluator.py`:
```python
@staticmethod
def custom_metric(response: str, context: List[str]) -> float:
    # Implement your metric
    return score
```

### Different Language Models
Update model configuration in `src/config.py` and ensure compatibility with the tokenizer.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## License

This project is developed for educational purposes as part of a RAG application assignment.

## Acknowledgments

- **Research Papers**: Authors of the transformer-based language model papers
- **Libraries**: LangChain, FAISS, Transformers, RAGAS, and other open-source libraries
- **Models**: Hugging Face model providers

---

For questions or issues, please refer to the troubleshooting section or create an issue in the repository. 