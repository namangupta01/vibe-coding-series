"""
Configuration settings for the RAG application.

This file contains all the important settings that control how our RAG application works.
Think of this as the "control panel" for the entire system - we can change these values
to modify how the application behaves without changing the actual code.

RAG = Retrieval-Augmented Generation
- Retrieval: Finding relevant information from documents
- Augmented: Using that information to help
- Generation: Creating responses to questions
"""
import os
from typing import List, Dict

class Config:
    """
    Configuration class for RAG application settings.
    
    This class holds all the important settings for our application.
    By keeping all settings in one place, it's easy to modify the behavior
    of the entire system by just changing values here.
    """
    
    # PDF URLs to download and process
    # These are the 5 research papers we want our bot to learn from
    # We use arXiv links because they provide free access to academic papers
    PDF_URLS: List[str] = [
        "https://arxiv.org/pdf/1706.03762.pdf",  # The famous "Attention Is All You Need" paper (Transformer)
        "https://arxiv.org/pdf/1810.04805.pdf",  # BERT - Bidirectional Encoder Representations
        "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3 - The large language model that started the AI revolution
        "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa - An improved version of BERT
        "https://arxiv.org/pdf/1910.10683.pdf",  # T5 - Text-to-Text Transfer Transformer
    ]
    
    # PDF names for local storage
    # These are the names we'll use to save the files on our computer
    # We use simple names instead of the complex arXiv URLs
    PDF_NAMES: List[str] = [
        "attention_is_all_you_need.pdf",  # The Transformer paper
        "bert.pdf",                       # BERT paper
        "gpt3.pdf",                      # GPT-3 paper
        "roberta.pdf",                   # RoBERTa paper
        "t5.pdf"                         # T5 paper
    ]
    
    # Model configurations
    # These specify which AI models we'll use for different tasks
    
    # EMBEDDING_MODEL: Converts text into numbers (vectors) for similarity search
    # "all-MiniLM-L6-v2" is a good balance of speed and accuracy
    # Smaller models = faster but less accurate, Larger models = slower but more accurate
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # LLM_MODEL: The language model that generates responses to questions
    # DialoGPT is designed for conversations, so it's good for our chatbot
    LLM_MODEL: str = "microsoft/DialoGPT-medium"
    
    # Text processing settings
    # When we read PDFs, we need to break the text into smaller pieces (chunks)
    # This is because AI models can only process limited amounts of text at once
    
    # CHUNK_SIZE: How many characters to put in each chunk
    # 1000 characters is roughly 150-200 words
    CHUNK_SIZE: int = 1000
    
    # CHUNK_OVERLAP: How many characters should overlap between chunks
    # Overlap helps ensure we don't cut sentences or ideas in half
    # 200 characters overlap means each chunk shares some text with the next chunk
    CHUNK_OVERLAP: int = 200
    
    # Vector database settings
    # The vector database stores our document chunks as numerical vectors
    # This allows us to find similar text quickly using math instead of keywords
    VECTOR_DB_PATH: str = "./data/vector_store"  # Where to save the vector database
    
    # Conversation memory settings
    # Our chatbot remembers previous conversations to provide better context
    # MEMORY_SIZE: How many previous question-answer pairs to remember
    # 4 means it remembers the last 4 conversations (8 total messages)
    MEMORY_SIZE: int = 4
    
    # Paths - where to save different types of files
    # Using relative paths (starting with "./") means folders are created in the project directory
    DATA_DIR: str = "./data"           # Main data folder
    PDF_DIR: str = "./data/pdfs"       # Where to save downloaded PDF files
    REPORTS_DIR: str = "./reports"     # Where to save evaluation reports
    
    # Evaluation settings
    # RAGAS is a framework for evaluating RAG systems
    # These metrics help us measure how well our system is working
    EVALUATION_METRICS: List[str] = [
        "faithfulness",      # Are the answers factually correct based on the documents?
        "answer_relevancy",  # Do the answers actually address the questions asked?
        "context_precision", # Is the retrieved information relevant to the question?
        "context_recall"     # Did we retrieve all the relevant information available?
    ]
    
    # Test questions for evaluation
    # These are the questions we'll use to test how well our system works
    # We designed them to cover different types of knowledge about AI/ML
    TEST_QUESTIONS: List[str] = [
        # Basic architecture question - tests understanding of key concepts
        "What is the Transformer architecture and how does it work?",
        
        # Technical mechanism question - tests detailed technical knowledge
        "Explain the concept of self-attention mechanism in neural networks.",
        
        # Comparison question - tests ability to distinguish between models
        "What are the key differences between BERT and GPT models?",
        
        # Specific technical detail - tests understanding of implementation details
        "How does positional encoding work in Transformer models?",
        
        # Historical significance - tests contextual understanding
        "What is the significance of the 'Attention is All You Need' paper?",
        
        # Training methodology - tests understanding of learning processes
        "Describe the training process of BERT and its masked language modeling objective.",
        
        # Practical applications - tests understanding of real-world use
        "What are the advantages of using pre-trained language models like GPT-3?",
        
        # Model improvements - tests understanding of evolutionary improvements
        "How does RoBERTa improve upon the original BERT model?",
        
        # Unified framework understanding - tests grasp of different approaches
        "What is the T5 model and how does it approach text-to-text transfer?",
        
        # Comparative analysis - tests ability to synthesize across multiple papers
        "Compare and contrast different attention mechanisms used in modern NLP models."
    ]
    
    @classmethod
    def create_directories(cls):
        """
        Create necessary directories if they don't exist.
        
        This is a utility function that makes sure all the folders we need
        actually exist on the computer before we try to save files in them.
        
        @classmethod means this function belongs to the class itself, not to
        any specific instance of the class. We can call it like Config.create_directories()
        """
        # List of all directories we need for the application to work
        directories = [cls.DATA_DIR, cls.PDF_DIR, cls.REPORTS_DIR]
        
        # Loop through each directory and create it if it doesn't exist
        for directory in directories:
            # os.makedirs creates the directory
            # exist_ok=True means "don't throw an error if the directory already exists"
            os.makedirs(directory, exist_ok=True) 