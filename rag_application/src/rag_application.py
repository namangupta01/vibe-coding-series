"""
Main RAG Application that integrates all components.

This is the "conductor" of our RAG orchestra! It brings together all the different
pieces and makes them work together harmoniously:

1. PDF PROCESSOR: Downloads and processes research papers
2. VECTOR STORE: Creates searchable database of document chunks  
3. CONVERSATIONAL BOT: Provides intelligent chat interface
4. EVALUATOR: Measures how well the system works

The application handles the complete workflow:
- Setup: Download PDFs → Extract text → Create vector database → Initialize bot
- Chat: Interactive conversations with the bot
- Evaluation: Test the system with predefined questions
- Reporting: Generate comprehensive reports

Think of this as the "main control room" that coordinates everything.
"""
import logging  # For tracking what happens
import os  # For file and directory operations
from typing import Dict, List, Any, Optional  # For type hints
from rich.console import Console  # For beautiful terminal output
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn  # For progress bars
from rich.table import Table  # For displaying data in tables
from rich.panel import Panel  # For highlighted text boxes

from .config import Config  # Configuration settings
from .pdf_processor import PDFProcessor  # For handling PDF files
from .vector_store import VectorStore  # For document search
from .conversational_bot import ConversationalBot  # For chat functionality
from .evaluator import RAGEvaluator  # For testing system performance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGApplication:
    """
    Main RAG Application orchestrating all components.
    
    This class is like the "project manager" of our RAG system. It:
    - Coordinates setup of all components
    - Manages the interaction between different parts
    - Provides user interfaces (chat, evaluation, etc.)
    - Tracks the overall state of the system
    
    Everything flows through this class to ensure proper coordination.
    """
    
    def __init__(self):
        """
        Initialize the RAG application.
        
        This sets up the basic structure but doesn't actually load
        any heavy components yet. That happens later in setup().
        """
        self.config = Config()
        
        # Initialize Rich console for beautiful terminal output
        self.console = Console()
        
        # Initialize our main components (but don't load them yet)
        self.pdf_processor = PDFProcessor()    # For handling PDF files
        self.vector_store = VectorStore()      # For document search
        self.bot = None                        # Chatbot (created during setup)
        self.evaluator = None                  # Evaluation system (created during setup)
        
        # Track the application state
        self.documents = None              # Processed document chunks
        self.is_setup_complete = False     # Whether setup has been completed
        
        logger.info("RAG Application initialized")
    
    def setup(self) -> None:
        """
        Complete setup process: download PDFs, create vector store, initialize bot.
        
        This is the "grand setup" that prepares everything:
        1. Download research papers from the internet
        2. Extract text and break it into chunks
        3. Create vector database for fast searching
        4. Initialize the conversational bot
        5. Set up evaluation system
        
        This might take 5-10 minutes the first time!
        """
        self.console.print("[bold blue]Setting up RAG Application...[/bold blue]")
        
        # Use Rich progress bars to show what's happening
        with Progress(
            SpinnerColumn(),              # Spinning animation
            *Progress.get_default_columns(),  # Standard progress columns
            TimeElapsedColumn(),          # Show elapsed time
            console=self.console,         # Where to display
        ) as progress:
            
            # Step 1: Download PDFs from arXiv
            task1 = progress.add_task("[cyan]Downloading PDFs...", total=100)
            downloaded_files = self.pdf_processor.download_pdfs()
            progress.update(task1, completed=100)
            
            # Check if we got any files
            if not downloaded_files:
                raise RuntimeError("Failed to download any PDF files")
            
            # Step 2: Extract text and create document chunks
            task2 = progress.add_task("[cyan]Processing PDFs to documents...", total=100)
            self.documents = self.pdf_processor.process_pdfs_to_documents(downloaded_files)
            progress.update(task2, completed=100)
            
            # Check if we got any documents
            if not self.documents:
                raise RuntimeError("Failed to process any documents")
            
            # Step 3: Create vector store (convert text to searchable vectors)
            task3 = progress.add_task("[cyan]Creating vector store...", total=100)
            self.vector_store.create_vector_store(self.documents)
            progress.update(task3, completed=100)
            
            # Step 4: Save vector store to disk (so we don't have to recreate it)
            task4 = progress.add_task("[cyan]Saving vector store...", total=100)
            self.vector_store.save_vector_store()
            progress.update(task4, completed=100)
            
            # Step 5: Initialize conversational bot with access to vector store
            task5 = progress.add_task("[cyan]Initializing conversational bot...", total=100)
            self.bot = ConversationalBot(self.vector_store)
            progress.update(task5, completed=100)
            
            # Step 6: Initialize evaluation system
            task6 = progress.add_task("[cyan]Initializing evaluator...", total=100)
            self.evaluator = RAGEvaluator(self.bot)
            progress.update(task6, completed=100)
        
        # Mark setup as complete
        self.is_setup_complete = True
        self.console.print("[bold green]✅ RAG Application setup complete![/bold green]")
        
        # Show a nice summary of what was accomplished
        self._display_setup_summary()
    
    def load_existing_setup(self) -> bool:
        """
        Load existing vector store and initialize components.
        
        This is much faster than full setup! If we've already done
        the heavy work (downloading, processing, creating vectors),
        we can just load the saved results.
        
        Returns:
            True if successful, False if no existing setup found
        """
        try:
            self.console.print("[yellow]Loading existing setup...[/yellow]")
            
            # Check if we have a saved vector store
            if not os.path.exists(self.config.VECTOR_DB_PATH):
                return False  # No existing setup found
            
            # Load the vector store from disk
            self.vector_store.load_vector_store()
            
            # Initialize bot and evaluator with loaded data
            self.bot = ConversationalBot(self.vector_store)
            self.evaluator = RAGEvaluator(self.bot)
            
            # Mark as complete
            self.is_setup_complete = True
            self.console.print("[bold green]✅ Existing setup loaded successfully![/bold green]")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing setup: {str(e)}")
            return False
    
    def _display_setup_summary(self):
        """
        Display a summary of the setup process.
        
        Shows useful statistics about what was accomplished:
        - How many documents were processed
        - Vector store information
        - Model details
        """
        if not self.is_setup_complete:
            return  # Nothing to show yet
        
        # Get statistics from different components
        doc_stats = self.pdf_processor.get_document_stats(self.documents)
        vector_stats = self.vector_store.get_vector_store_stats()
        
        # Create a beautiful table using Rich
        table = Table(title="RAG Application Setup Summary", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")
        
        # Add rows with key information
        table.add_row("PDFs Processed", str(len(self.config.PDF_NAMES)))
        table.add_row("Total Documents", str(doc_stats['total_documents']))
        table.add_row("Total Characters", f"{doc_stats['total_characters']:,}")
        table.add_row("Average Chunk Size", f"{doc_stats['average_chunk_size']:.0f}")
        table.add_row("Vector Store Size", str(vector_stats['total_vectors']))
        table.add_row("Vector Dimension", str(vector_stats['vector_dimension']))
        table.add_row("Embedding Model", vector_stats['embedding_model'])
        table.add_row("Memory Size", str(self.config.MEMORY_SIZE))
        
        # Display the table
        self.console.print(table)
    
    def chat_interactive(self):
        """
        Start an interactive chat session with the RAG bot.
        
        This creates a terminal-based chat interface where users can:
        - Ask questions and get AI responses
        - See which documents were used
        - View conversation statistics
        - Clear memory or exit
        
        It's like having a conversation with an AI research assistant!
        """
        # Make sure setup is complete
        if not self.is_setup_complete:
            self.console.print("[red]Please run setup() first![/red]")
            return
        
        # Display welcome message
        self.console.print(Panel(
            "[bold blue]RAG Application Interactive Chat[/bold blue]\n"
            "Type 'quit', 'exit', or 'q' to end the session.\n"
            "Type 'clear' to clear conversation history.\n"
            "Type 'stats' to see conversation statistics.",
            title="Chat Session Started"
        ))
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break  # Exit the chat
                elif user_input.lower() == 'clear':
                    self.bot.clear_conversation_history()
                    self.console.print("[yellow]Conversation history cleared.[/yellow]")
                    continue
                elif user_input.lower() == 'stats':
                    self._display_chat_stats()
                    continue
                elif not user_input:
                    continue  # Skip empty input
                
                # Process the question and get bot's response
                response = self.bot.chat(user_input)
                
                # Display the response
                self.console.print(f"\n🤖 [bold green]Bot:[/bold green] {response['response']}")
                
                # Show which sources were used (if any)
                if response['retrieved_sources']:
                    sources = ", ".join(response['retrieved_sources'])
                    self.console.print(f"[dim]📚 Sources: {sources}[/dim]")
                
                # Show memory usage
                memory_info = response['memory_stats']
                self.console.print(f"[dim]💭 Memory: {memory_info['memory_usage']}[/dim]")
                
            except KeyboardInterrupt:
                break  # Exit if user presses Ctrl+C
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
        
        self.console.print("\n[blue]Chat session ended. Goodbye![/blue]")
    
    def _display_chat_stats(self):
        """
        Display current chat session statistics.
        
        Shows a table with:
        - All questions asked
        - Response lengths
        - Sources used
        """
        if not self.bot:
            return
        
        # Get conversation history
        history = self.bot.get_conversation_history()
        if not history:
            self.console.print("[yellow]No conversation history available.[/yellow]")
            return
        
        # Create statistics table
        table = Table(title="Conversation Statistics", show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Question", style="white", max_width=50)
        table.add_column("Response Length", style="green", width=15)
        table.add_column("Sources", style="yellow", width=20)
        
        # Add each conversation to the table
        for i, interaction in enumerate(history, 1):
            # Truncate long questions for display
            question = interaction['user_input'][:47] + "..." if len(interaction['user_input']) > 50 else interaction['user_input']
            
            # Count response length in words
            response_len = str(len(interaction['bot_response'].split()))
            
            # Count number of sources used
            sources = str(len(interaction.get('context', [])))
            
            table.add_row(str(i), question, response_len, sources)
        
        self.console.print(table)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation process.
        
        This tests our RAG system comprehensively:
        1. Asks all test questions
        2. Evaluates responses using RAGAS metrics
        3. Applies custom quality metrics
        4. Generates overall performance scores
        5. Saves results for reporting
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Make sure setup is complete
        if not self.is_setup_complete:
            self.console.print("[red]Please run setup() first![/red]")
            return {}
        
        self.console.print("[bold blue]Running RAG Application Evaluation...[/bold blue]")
        
        # Run evaluation with progress tracking
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Evaluating RAG application...", total=100)
            
            # This is where the heavy work happens
            results = self.evaluator.run_full_evaluation()
            progress.update(task, completed=100)
        
        # Display results in a nice format
        self.evaluator.print_evaluation_summary()
        
        # Save results to file for later use
        self.evaluator.save_evaluation_results()
        
        return results
    
    def answer_test_questions(self) -> List[Dict[str, Any]]:
        """
        Answer all test questions without full evaluation.
        
        This is a lighter version of evaluation - just gets answers
        to all test questions without running the heavy metrics.
        Good for quick testing or demos.
        
        Returns:
            List of question-answer pairs with sources
        """
        # Make sure setup is complete
        if not self.is_setup_complete:
            self.console.print("[red]Please run setup() first![/red]")
            return []
        
        self.console.print("[bold blue]Answering Test Questions...[/bold blue]")
        
        results = []
        
        # Process each test question with progress tracking
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            
            task = progress.add_task("[cyan]Processing questions...", total=len(self.config.TEST_QUESTIONS))
            
            # Ask each question and collect responses
            for i, question in enumerate(self.config.TEST_QUESTIONS):
                response = self.bot.chat(question)
                results.append({
                    'question': question,
                    'answer': response['response'],
                    'sources': response.get('retrieved_sources', [])
                })
                progress.update(task, advance=1)
        
        # Display results in a nice table
        table = Table(title="Test Questions and Answers", show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Question", style="white", max_width=40)
        table.add_column("Answer", style="green", max_width=60)
        
        for i, result in enumerate(results, 1):
            # Truncate long text for display
            question = result['question'][:37] + "..." if len(result['question']) > 40 else result['question']
            answer = result['answer'][:57] + "..." if len(result['answer']) > 60 else result['answer']
            
            table.add_row(str(i), question, answer)
        
        self.console.print(table)
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns detailed information about:
        - Setup status
        - Configuration settings
        - Document statistics
        - Vector store status
        - Conversation history
        
        Returns:
            Dictionary with complete system information
        """
        # Basic information (always available)
        info = {
            'setup_complete': self.is_setup_complete,
            'config': {
                'pdf_count': len(self.config.PDF_NAMES),
                'embedding_model': self.config.EMBEDDING_MODEL,
                'llm_model': self.config.LLM_MODEL,
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'memory_size': self.config.MEMORY_SIZE
            }
        }
        
        # Additional information if setup is complete
        if self.is_setup_complete:
            info.update({
                'document_stats': self.pdf_processor.get_document_stats(self.documents) if self.documents else {},
                'vector_store_stats': self.vector_store.get_vector_store_stats(),
                'conversation_count': len(self.bot.get_conversation_history()) if self.bot else 0
            })
        
        return info
    
    def reset_application(self):
        """
        Reset the application state and clean up resources.
        
        This "starts over" by:
        - Clearing conversation history
        - Resetting all components
        - Optionally deleting saved data
        
        Use this to start fresh or clean up.
        """
        self.console.print("[yellow]Resetting RAG Application...[/yellow]")
        
        # Clear conversation history if bot exists
        if self.bot:
            self.bot.clear_conversation_history()
        
        # Reset all components to initial state
        self.bot = None
        self.evaluator = None
        self.documents = None
        self.is_setup_complete = False
        
        # Optionally delete saved vector store
        try:
            self.vector_store.delete_vector_store()
        except Exception as e:
            logger.warning(f"Could not delete vector store: {str(e)}")
        
        self.console.print("[green]Application reset complete.[/green]")
    
    def cleanup(self):
        """
        Clean up resources and temporary files.
        
        This is called when the application shuts down to
        ensure proper cleanup of resources.
        """
        logger.info("Cleaning up RAG application resources")
        
        # Any cleanup operations would go here
        # For now, just log that cleanup is complete
        logger.info("Cleanup complete") 