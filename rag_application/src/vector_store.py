"""
Vector store module for creating embeddings and managing FAISS vector database.

This module is the "search engine" of our RAG system. It works like this:

1. EMBEDDINGS: Convert text into numbers (vectors) that represent meaning
   - Similar texts have similar numbers
   - This lets us find related content using math instead of keywords

2. VECTOR DATABASE: Store all our document chunks as vectors
   - Uses FAISS (Facebook AI Similarity Search) for fast searching
   - Can quickly find the most similar documents to a question

3. SIMILARITY SEARCH: When someone asks a question:
   - Convert the question to a vector
   - Find document vectors that are most similar
   - Return those documents as context for answering

Think of this like a librarian who has read every book and can instantly
find the most relevant passages for any question you ask.
"""
import os
import pickle  # For saving Python objects to files
import logging  # For tracking what happens
from typing import List, Dict, Tuple, Optional  # For type hints
import numpy as np  # For working with numbers and arrays
from sentence_transformers import SentenceTransformer  # For creating embeddings
import faiss  # Facebook's fast similarity search library
from langchain.schema import Document  # For storing text with metadata
from langchain_community.vectorstores import FAISS  # LangChain wrapper for FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # For converting text to vectors

from .config import Config  # Our configuration settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector embeddings and FAISS-based similarity search.
    
    This class handles everything related to:
    - Converting text to numerical vectors (embeddings)
    - Storing those vectors in a searchable database
    - Finding similar text when given a question
    
    It's like having a super-fast librarian who can instantly find
    relevant information from thousands of documents.
    """
    
    def __init__(self):
        """
        Initialize the vector store with embedding model.
        
        This sets up our "text-to-numbers" converter and prepares
        everything we need for similarity search.
        """
        self.config = Config()
        
        # Initialize embedding model
        # This is the AI model that converts text into vectors (lists of numbers)
        # These vectors capture the "meaning" of the text in mathematical form
        logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,     # Which model to use
            model_kwargs={'device': 'cpu'},             # Use CPU (change to 'cuda' for GPU)
            encode_kwargs={'normalize_embeddings': False}  # Don't normalize the vectors
        )
        
        # These will be set up later when we create or load a vector store
        self.vector_store = None    # The actual FAISS database
        self.documents = None       # The original documents (for reference)
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a FAISS vector store from documents.
        
        This is where the magic happens! We take all our document chunks
        and convert them into a searchable vector database.
        
        Steps:
        1. Take each document chunk
        2. Convert it to a vector using our embedding model
        3. Store all vectors in FAISS for fast searching
        
        Args:
            documents: List of Document objects to embed (turn into vectors)
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        # Store documents for later retrieval
        # We keep the original documents so we can return the actual text, not just vectors
        self.documents = documents
        
        # Create FAISS vector store
        # This does the heavy lifting: converts all documents to vectors and indexes them
        self.vector_store = FAISS.from_documents(
            documents=documents,    # The documents to convert
            embedding=self.embeddings  # The model to use for conversion
        )
        
        logger.info("Vector store created successfully")
    
    def save_vector_store(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        This saves our vector database to the computer's hard drive
        so we don't have to recreate it every time we run the program.
        
        Args:
            path: Optional custom path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        # Use default path if none provided
        save_path = path or self.config.VECTOR_DB_PATH
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info(f"Saving vector store to {save_path}")
        
        # Save the FAISS index and related data
        self.vector_store.save_local(save_path)
        
        # Also save the original documents separately
        # This lets us get back the full document text when needed
        documents_path = os.path.join(save_path, "documents.pkl")
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info("Vector store saved successfully")
    
    def load_vector_store(self, path: Optional[str] = None) -> None:
        """
        Load a vector store from disk.
        
        This loads a previously saved vector database from the hard drive.
        Much faster than creating a new one from scratch!
        
        Args:
            path: Optional custom path to load the vector store from
        """
        # Use default path if none provided
        load_path = path or self.config.VECTOR_DB_PATH
        
        # Check if the vector store exists
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        logger.info(f"Loading vector store from {load_path}")
        
        # Load the FAISS vector store
        self.vector_store = FAISS.load_local(
            load_path,                              # Where to load from
            self.embeddings,                        # The embedding model (needed for queries)
            allow_dangerous_deserialization=True   # Allow loading pickled data (needed for FAISS)
        )
        
        # Load the original documents if they exist
        documents_path = os.path.join(load_path, "documents.pkl")
        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        
        logger.info("Vector store loaded successfully")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search to retrieve relevant documents.
        
        This is the core function that finds relevant information!
        
        How it works:
        1. Convert the question to a vector
        2. Find the k most similar document vectors
        3. Return the original documents for those vectors
        
        Args:
            query: Search query (the question someone asked)
            k: Number of documents to retrieve (how many results to return)
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        logger.debug(f"Performing similarity search for: {query[:50]}...")
        
        # Use FAISS to find the most similar documents
        results = self.vector_store.similarity_search(query, k=k)
        
        logger.debug(f"Retrieved {len(results)} documents")
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Same as similarity_search, but also returns how similar each document is.
        The score tells us how confident we are that this document is relevant.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (document, similarity_score)
            Lower scores = more similar (closer in vector space)
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        logger.debug(f"Performing similarity search with scores for: {query[:50]}...")
        
        # Get results with similarity scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        logger.debug(f"Retrieved {len(results)} documents with scores")
        return results
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None) -> object:
        """
        Get a retriever object for use with LangChain.
        
        A "retriever" is a LangChain concept - it's an object that can
        find relevant documents given a query. This is useful for
        integrating with other LangChain components.
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            LangChain retriever object
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        # Set up default search parameters
        default_kwargs = {"k": 5}  # Return 5 documents by default
        if search_kwargs:
            default_kwargs.update(search_kwargs)  # Override with user preferences
        
        # Create and return the retriever
        return self.vector_store.as_retriever(search_kwargs=default_kwargs)
    
    def get_vector_store_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        This gives us useful information about our vector database:
        - How many vectors we have
        - What dimension they are
        - Which documents they came from
        
        Returns:
            Dictionary with vector store statistics
        """
        if self.vector_store is None:
            return {"status": "No vector store loaded"}
        
        # Get basic FAISS statistics
        stats = {
            "total_vectors": self.vector_store.index.ntotal,    # How many document chunks
            "vector_dimension": self.vector_store.index.d,     # Size of each vector
            "embedding_model": self.config.EMBEDDING_MODEL,    # Which model we used
        }
        
        # Add document statistics if available
        if self.documents:
            stats.update({
                "total_documents": len(self.documents),
                "documents_per_source": {}
            })
            
            # Count documents per source PDF
            for doc in self.documents:
                source = doc.metadata.get('source', 'unknown')
                stats["documents_per_source"][source] = stats["documents_per_source"].get(source, 0) + 1
        
        return stats
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        This lets us add more documents to our database without
        recreating the entire thing from scratch.
        
        Args:
            documents: List of new documents to add
        """
        if not documents:
            return  # Nothing to add
        
        if self.vector_store is None:
            # If we don't have a vector store yet, create one
            self.create_vector_store(documents)
        else:
            # Add to existing vector store
            logger.info(f"Adding {len(documents)} new documents to vector store")
            self.vector_store.add_documents(documents)
            
            # Update our stored documents
            if self.documents:
                self.documents.extend(documents)
            else:
                self.documents = documents
        
        logger.info("Documents added successfully")
    
    def delete_vector_store(self, path: Optional[str] = None) -> None:
        """
        Delete the vector store from disk.
        
        This removes the saved vector database from the hard drive.
        Useful for cleaning up or starting fresh.
        
        Args:
            path: Optional custom path of the vector store to delete
        """
        import shutil  # For deleting directories
        
        # Use default path if none provided
        delete_path = path or self.config.VECTOR_DB_PATH
        
        # Delete the directory if it exists
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)  # Remove the entire directory tree
            logger.info(f"Vector store deleted from {delete_path}")
        else:
            logger.warning(f"Vector store not found at {delete_path}")
        
        # Reset instance variables
        self.vector_store = None
        self.documents = None 