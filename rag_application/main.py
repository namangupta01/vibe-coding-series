"""
Main entry point for the RAG Application.
Provides command-line interface for setup, interaction, evaluation, and reporting.

This is the file you run to start the RAG application! It provides a command-line
interface that lets you:

1. SETUP: Download PDFs, create vector database, initialize everything
2. CHAT: Have interactive conversations with the AI bot
3. EVALUATE: Test the system's performance with metrics
4. REPORT: Generate PDF reports of evaluation results
5. DEMO: Run the complete workflow automatically

Example usage:
    python main.py --setup        # Set up the application
    python main.py --chat         # Start chatting
    python main.py --evaluate     # Run evaluation
    python main.py --demo         # Do everything automatically

Think of this as the "front door" to our RAG application.
"""
import argparse  # For parsing command-line arguments
import logging   # For tracking what happens
import sys       # For system operations
from pathlib import Path  # For handling file paths

# Add src to path so we can import our modules
# This tells Python where to find our custom code
sys.path.append(str(Path(__file__).parent / "src"))

# Import our main application components
from src.rag_application import RAGApplication      # The main application
from src.report_generator import RAGReportGenerator  # For creating PDF reports

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main application entry point.
    
    This function:
    1. Parses command-line arguments to see what the user wants
    2. Creates the RAG application instance
    3. Executes the requested operation (setup, chat, evaluate, etc.)
    4. Handles any errors that occur
    """
    # Set up command-line argument parsing
    # This defines all the options users can choose from
    parser = argparse.ArgumentParser(description="RAG Application - Retrieval-Augmented Generation System")
    
    # Define all available command-line options
    parser.add_argument("--setup", action="store_true", 
                       help="Setup the RAG application (download PDFs, create vector store)")
    parser.add_argument("--chat", action="store_true", 
                       help="Start interactive chat session")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Run full evaluation with RAGAS and custom metrics")
    parser.add_argument("--questions", action="store_true", 
                       help="Answer test questions without evaluation")
    parser.add_argument("--report", action="store_true", 
                       help="Generate PDF report (requires evaluation first)")
    parser.add_argument("--demo", action="store_true", 
                       help="Run complete demonstration (setup + evaluate + report)")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset application and clean up resources")
    parser.add_argument("--info", action="store_true", 
                       help="Display system information")
    
    # Parse the arguments the user provided
    args = parser.parse_args()
    
    # If no arguments provided, show help message
    # This helps users understand what options are available
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Initialize the main RAG application
    app = RAGApplication()
    
    try:
        # Handle reset first (before doing anything else)
        if args.reset:
            app.reset_application()
            return
        
        # Handle info request (show system status)
        if args.info:
            info = app.get_system_info()
            print("\n" + "="*50)
            print("RAG APPLICATION SYSTEM INFO")
            print("="*50)
            print(f"Setup Complete: {info['setup_complete']}")
            
            # Show additional info if setup is complete
            if info['setup_complete']:
                print(f"Documents: {info['document_stats'].get('total_documents', 0)}")
                print(f"Vector Store: {info['vector_store_stats'].get('total_vectors', 0)} vectors")
                print(f"Conversations: {info.get('conversation_count', 0)}")
            print("="*50)
            return
        
        # Try to load existing setup first (faster than creating new)
        if not app.load_existing_setup():
            # If no existing setup found, create new one if requested
            if args.setup or args.demo:
                app.setup()
            else:
                print("❌ No existing setup found. Please run with --setup first.")
                return
        
        # Handle different operations based on user's choice
        
        if args.chat:
            # Start interactive chat session
            app.chat_interactive()
        
        elif args.questions:
            # Answer test questions without full evaluation
            results = app.answer_test_questions()
            print(f"\n✅ Answered {len(results)} test questions successfully!")
        
        elif args.evaluate or args.demo:
            # Run comprehensive evaluation
            print("🔍 Running comprehensive evaluation...")
            evaluation_results = app.run_evaluation()
            
            # Generate report if requested
            if args.report or args.demo:
                print("📝 Generating PDF report...")
                report_generator = RAGReportGenerator()
                system_info = app.get_system_info()
                report_path = report_generator.generate_report(evaluation_results, system_info)
                print(f"📄 Report generated: {report_path}")
        
        elif args.report:
            # User wants report but hasn't run evaluation
            print("❌ Cannot generate report without evaluation results.")
            print("Please run with --evaluate first, or use --demo for complete workflow.")
        
        # Clean up resources when done
        app.cleanup()
        
    except KeyboardInterrupt:
        # User pressed Ctrl+C to stop
        print("\n\n👋 Application interrupted by user.")
    except Exception as e:
        # Something went wrong
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Error: {str(e)}")
    finally:
        # Always clean up, even if something goes wrong
        print("🔄 Cleaning up resources...")

def run_demo():
    """
    Run a complete demonstration of the RAG application.
    
    This function provides a full walkthrough of the system:
    1. Sets up everything from scratch
    2. Answers some sample questions
    3. Runs comprehensive evaluation
    4. Generates a PDF report
    
    It's like a guided tour of all the system's capabilities!
    """
    print("🚀 Starting RAG Application Demo")
    print("="*50)
    
    # Create application instance
    app = RAGApplication()
    
    try:
        # Step 1: Setup the application
        print("\n📋 Step 1: Setting up RAG Application...")
        if not app.load_existing_setup():
            # No existing setup, create new one
            app.setup()
        
        # Step 2: Demo some conversations
        print("\n💬 Step 2: Answering sample questions...")
        sample_questions = [
            "What is the Transformer architecture?",
            "How does BERT differ from GPT?",
            "What is self-attention in neural networks?"
        ]
        
        # Ask each sample question and show response
        for question in sample_questions:
            print(f"\nQ: {question}")
            response = app.bot.chat(question)
            # Show truncated response for demo purposes
            print(f"A: {response['response'][:200]}...")
        
        # Step 3: Run comprehensive evaluation
        print("\n🔍 Step 3: Running evaluation...")
        evaluation_results = app.run_evaluation()
        
        # Step 4: Generate PDF report
        print("\n📝 Step 4: Generating report...")
        report_generator = RAGReportGenerator()
        system_info = app.get_system_info()
        report_path = report_generator.generate_report(evaluation_results, system_info)
        
        print("\n✅ Demo completed successfully!")
        print(f"📄 Report available at: {report_path}")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Demo error: {str(e)}")
    finally:
        # Always clean up
        app.cleanup()

# This ensures the code only runs when the file is executed directly
# (not when it's imported as a module)
if __name__ == "__main__":
    main() 