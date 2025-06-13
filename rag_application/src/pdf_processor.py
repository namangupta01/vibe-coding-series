"""
PDF processing module for downloading and extracting text from research papers.

This module handles everything related to working with PDF files:
1. Downloading PDFs from the internet (arXiv papers)
2. Extracting text from PDF files
3. Cleaning up the extracted text (removing weird characters, fixing formatting)
4. Breaking the text into smaller chunks that our AI models can process

Think of this as the "librarian" of our system - it gets the books (PDFs),
reads them, cleans up the text, and organizes it into manageable pieces.
"""
import os
import requests  # For downloading files from the internet
import PyPDF2   # For reading PDF files
from io import BytesIO  # For handling file data in memory
from typing import List, Dict, Tuple  # For type hints (making code clearer)
from tqdm import tqdm  # For showing progress bars during downloads
import logging  # For recording what happens (like a diary for the program)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text intelligently
from langchain.schema import Document  # For storing text with metadata

from .config import Config  # Import our configuration settings

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF downloading, text extraction, and preprocessing.
    
    This class is like a Swiss Army knife for PDF files. It can:
    - Download PDFs from URLs
    - Extract text from PDF files
    - Clean up messy text
    - Split text into chunks for AI processing
    """
    
    def __init__(self):
        """
        Initialize the PDF processor.
        
        This function runs when we create a new PDFProcessor object.
        It sets up everything we need to process PDFs.
        """
        # Get our configuration settings
        self.config = Config()
        
        # Set up the text splitter
        # This tool breaks long text into smaller pieces (chunks)
        # It's "recursive" because it tries different ways to split text nicely
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,        # How big each chunk should be
            chunk_overlap=self.config.CHUNK_OVERLAP,  # How much chunks should overlap
            length_function=len,                      # How to measure text length (by characters)
            separators=["\n\n", "\n", " ", ""]       # Where to prefer splitting (paragraphs, lines, words, characters)
        )
    
    def download_pdfs(self) -> Dict[str, str]:
        """
        Download PDFs from the configured URLs.
        
        This function goes to the internet, downloads each PDF file,
        and saves it to our local computer.
        
        Returns:
            Dict mapping PDF names to local file paths
            Example: {"bert.pdf": "./data/pdfs/bert.pdf"}
        """
        # Make sure the directories exist before we try to save files
        Config.create_directories()
        
        # Dictionary to keep track of which files we successfully downloaded
        downloaded_files = {}
        
        # Loop through each URL and filename pair
        for url, name in zip(self.config.PDF_URLS, self.config.PDF_NAMES):
            # Figure out where to save this file on our computer
            file_path = os.path.join(self.config.PDF_DIR, name)
            
            # Check if we already have this file
            if os.path.exists(file_path):
                logger.info(f"PDF {name} already exists, skipping download")
                downloaded_files[name] = file_path
                continue  # Skip to the next file
            
            try:
                # Try to download the file
                logger.info(f"Downloading {name} from {url}")
                
                # Send a request to download the file
                # stream=True means download in chunks instead of all at once
                response = requests.get(url, stream=True)
                
                # Check if the download was successful
                response.raise_for_status()  # Throws an error if something went wrong
                
                # Get the size of the file so we can show a progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                # Save the file to our computer
                with open(file_path, 'wb') as f:  # 'wb' means write binary (for PDF files)
                    # Create a progress bar so we can see download progress
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar:
                        # Download the file in chunks (8192 bytes at a time)
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # Make sure the chunk isn't empty
                                f.write(chunk)           # Save this chunk to the file
                                pbar.update(len(chunk))  # Update the progress bar
                
                # Remember that we successfully downloaded this file
                downloaded_files[name] = file_path
                logger.info(f"Successfully downloaded {name}")
                
            except Exception as e:
                # If something went wrong, log the error but keep trying other files
                logger.error(f"Failed to download {name}: {str(e)}")
                
        return downloaded_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        This function opens a PDF file and reads all the text from it.
        PDFs can be tricky because they're designed for visual display,
        not for extracting text, so we need special tools.
        
        Args:
            pdf_path: Path to the PDF file on our computer
            
        Returns:
            Extracted text as a single string
        """
        try:
            text = ""  # Start with empty text
            
            # Open the PDF file
            with open(pdf_path, 'rb') as file:  # 'rb' means read binary
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Loop through each page in the PDF
                for page_num in range(len(pdf_reader.pages)):
                    # Get the current page
                    page = pdf_reader.pages[page_num]
                    
                    # Extract text from this page and add it to our total text
                    text += page.extract_text() + "\n"
            
            # Clean up the text to make it nicer to work with
            text = self._clean_text(text)
            return text
            
        except Exception as e:
            # If something goes wrong, log the error and return empty text
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and formatting issues.
        
        When we extract text from PDFs, it often has weird formatting issues:
        - Extra spaces everywhere
        - Strange characters
        - Words split across lines with hyphens
        
        This function fixes these common problems.
        
        Args:
            text: Raw extracted text (messy)
            
        Returns:
            Cleaned text (much nicer)
        """
        # Remove excessive whitespace
        # ' '.join(text.split()) replaces any amount of whitespace with single spaces
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts (weird characters that shouldn't be there)
        text = text.replace('\x0c', '')  # Form feed characters (page breaks)
        text = text.replace('\xa0', ' ')  # Non-breaking spaces (replace with regular spaces)
        
        # Fix common word splitting issues
        # PDFs sometimes split words across lines with hyphens, like "atten-tion"
        text = text.replace('- ', '')  # Remove hyphens followed by spaces
        
        return text
    
    def process_pdfs_to_documents(self, pdf_files: Dict[str, str]) -> List[Document]:
        """
        Process PDF files into LangChain Document objects with chunks.
        
        This is the main function that takes our PDF files and turns them into
        a format that our AI system can work with. It:
        1. Extracts text from each PDF
        2. Splits the text into chunks
        3. Creates Document objects with metadata
        
        Args:
            pdf_files: Dictionary mapping PDF names to file paths
            
        Returns:
            List of Document objects (each chunk becomes one Document)
        """
        documents = []  # List to store all our document chunks
        
        # Process each PDF file
        for pdf_name, pdf_path in pdf_files.items():
            logger.info(f"Processing {pdf_name}")
            
            # Step 1: Extract text from the PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Check if we got any text
            if not text:
                logger.warning(f"No text extracted from {pdf_name}")
                continue  # Skip this file and move to the next one
            
            # Step 2: Create a Document object with the full text
            doc = Document(
                page_content=text,  # The actual text content
                metadata={          # Extra information about this document
                    'source': pdf_name,           # Which PDF this came from
                    'file_path': pdf_path,        # Where the PDF is stored
                    'total_length': len(text)     # How long the text is
                }
            )
            
            # Step 3: Split the document into smaller chunks
            # This is important because AI models can only process limited amounts of text
            chunks = self.text_splitter.split_documents([doc])
            
            # Step 4: Add chunk-specific metadata to each piece
            for i, chunk in enumerate(chunks):
                # Add information about which chunk this is
                chunk.metadata.update({
                    'chunk_index': i,                # Which chunk number (0, 1, 2, ...)
                    'total_chunks': len(chunks)      # How many total chunks from this PDF
                })
            
            # Add all chunks from this PDF to our documents list
            documents.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {pdf_name}")
        
        logger.info(f"Total documents created: {len(documents)}")
        return documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict:
        """
        Get statistics about the processed documents.
        
        This function analyzes our documents and gives us useful information
        like how many documents we have, how long they are, etc.
        
        Args:
            documents: List of Document objects to analyze
            
        Returns:
            Dictionary with document statistics
        """
        # Initialize our statistics
        stats = {
            'total_documents': len(documents),  # How many document chunks we have
            'total_characters': sum(len(doc.page_content) for doc in documents),  # Total amount of text
            'average_chunk_size': 0,            # Average size of each chunk
            'documents_per_source': {}          # How many chunks came from each PDF
        }
        
        # Calculate average chunk size (avoid division by zero)
        if documents:
            stats['average_chunk_size'] = stats['total_characters'] / len(documents)
            
            # Count how many document chunks came from each source PDF
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')  # Get the source PDF name
                # Increment the count for this source (start with 0 if we haven't seen it before)
                stats['documents_per_source'][source] = stats['documents_per_source'].get(source, 0) + 1
        
        return stats 