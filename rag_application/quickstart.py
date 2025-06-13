#!/usr/bin/env python3
"""
Quick Start Script for RAG Application
This script provides a simple way to get started with the RAG application.

This is the "beginner-friendly" version of the RAG application!
Instead of using command-line arguments, it provides a simple menu
that walks you through all the options.

Perfect for users who prefer interactive menus over command-line interfaces.
Just run this file and follow the prompts!

Features:
- Menu-driven interface (no complex commands to remember)
- Step-by-step guidance
- Built-in demo mode
- User-friendly error messages

Usage:
    python quickstart.py        # Start the interactive menu
    python quickstart.py --demo # Run a quick demo
"""

import sys
from pathlib import Path
import time  # For measuring how long things take

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

def print_header():
    """
    Print application header.
    
    This creates a nice welcome message that explains what
    the application does and makes users feel welcome.
    """
    print("🚀 RAG Application Quick Start")
    print("=" * 50)
    print("Retrieval-Augmented Generation System")
    print("Developed for transformer-based language model Q&A")
    print("=" * 50)

def main():
    """
    Main quick start function.
    
    This creates an interactive menu system that guides users
    through all the available options. Much friendlier than
    remembering command-line arguments!
    
    The menu keeps running until the user chooses to exit,
    making it easy to try multiple features.
    """
    print_header()
    
    try:
        # Import our modules (this might take a moment)
        from src.rag_application import RAGApplication
        from src.report_generator import RAGReportGenerator
        
        print("\n📋 Initializing RAG Application...")
        app = RAGApplication()
        
        # Try to load existing setup, otherwise guide user through setup
        print("🔍 Checking for existing setup...")
        if app.load_existing_setup():
            print("✅ Loaded existing setup!")
        else:
            print("📥 No existing setup found. Setting up from scratch...")
            print("⏳ This may take 5-10 minutes for first-time setup...")
            app.setup()
        
        print("\n🎯 RAG Application is ready!")
        
        # Display the main menu options
        print("\nAvailable operations:")
        print("1. Interactive Chat")           # Talk with the AI bot
        print("2. Answer Test Questions")      # Run through predefined questions
        print("3. Run Full Evaluation")       # Comprehensive testing
        print("4. Generate Report")          # Create PDF report
        print("5. System Information")       # Show current status
        print("6. Exit")                     # Quit the application
        
        # Main menu loop - keeps running until user exits
        while True:
            try:
                # Get user's choice
                choice = input("\nSelect an option (1-6): ").strip()
                
                # Handle each menu option
                if choice == "1":
                    # OPTION 1: Interactive Chat
                    print("\n💬 Starting interactive chat...")
                    print("Type 'quit' to return to main menu.")
                    app.chat_interactive()
                
                elif choice == "2":
                    # OPTION 2: Answer Test Questions
                    print("\n❓ Answering test questions...")
                    start_time = time.time()  # Track how long it takes
                    results = app.answer_test_questions()
                    elapsed = time.time() - start_time
                    print(f"\n✅ Completed {len(results)} questions in {elapsed:.2f} seconds")
                
                elif choice == "3":
                    # OPTION 3: Full Evaluation
                    print("\n🔍 Running comprehensive evaluation...")
                    print("⏳ This may take several minutes...")
                    start_time = time.time()
                    evaluation_results = app.run_evaluation()
                    elapsed = time.time() - start_time
                    print(f"\n✅ Evaluation completed in {elapsed:.2f} seconds")
                    
                    # Offer to generate report after evaluation
                    if input("\nGenerate PDF report? (y/n): ").lower().startswith('y'):
                        print("📝 Generating PDF report...")
                        report_generator = RAGReportGenerator()
                        system_info = app.get_system_info()
                        report_path = report_generator.generate_report(evaluation_results, system_info)
                        print(f"📄 Report saved to: {report_path}")
                
                elif choice == "4":
                    # OPTION 4: Generate Report
                    print("\n📝 Generating PDF report...")
                    print("Note: This requires evaluation to be run first.")
                    # This would need evaluation results stored somewhere
                    print("Please run evaluation first (option 3).")
                
                elif choice == "5":
                    # OPTION 5: System Information
                    print("\n📊 System Information:")
                    info = app.get_system_info()
                    print(f"  Setup Complete: {info['setup_complete']}")
                    
                    # Show detailed info if setup is complete
                    if info['setup_complete']:
                        print(f"  Documents: {info['document_stats'].get('total_documents', 0)}")
                        print(f"  Vector Store: {info['vector_store_stats'].get('total_vectors', 0)} vectors")
                        print(f"  Conversations: {info.get('conversation_count', 0)}")
                        print(f"  Embedding Model: {info['config'].get('embedding_model', 'N/A')}")
                        print(f"  LLM Model: {info['config'].get('llm_model', 'N/A')}")
                
                elif choice == "6":
                    # OPTION 6: Exit
                    print("\n👋 Goodbye!")
                    break
                
                else:
                    # Invalid choice
                    print("❌ Invalid choice. Please select 1-6.")
                
            except KeyboardInterrupt:
                # User pressed Ctrl+C
                print("\n\n👋 Returning to main menu...")
                continue
            except Exception as e:
                # Something went wrong
                print(f"❌ Error: {str(e)}")
                continue
        
        # Clean up when done
        app.cleanup()
        
    except ImportError as e:
        # Module import failed
        print(f"❌ Import Error: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        # Some other error occurred
        print(f"❌ Unexpected Error: {str(e)}")
        print("Please check the logs for more details.")

def demo_questions():
    """
    Demonstrate the system with a few sample questions.
    
    This is a quick way to see the system in action without
    going through the full menu. It asks a few interesting
    questions and shows how the bot responds.
    
    Perfect for getting a quick taste of what the system can do!
    """
    print("\n🎯 Quick Demo - Sample Questions")
    print("-" * 40)
    
    try:
        from src.rag_application import RAGApplication
        
        app = RAGApplication()
        
        # Make sure the system is set up
        if not app.load_existing_setup():
            print("❌ Please run setup first!")
            return
        
        # Ask some interesting demo questions
        demo_questions = [
            "What is the key innovation of the Transformer architecture?",
            "How does BERT's training differ from traditional language models?",
            "What makes GPT-3 particularly impressive in few-shot learning?"
        ]
        
        # Process each demo question
        for i, question in enumerate(demo_questions, 1):
            print(f"\n💬 Question {i}: {question}")
            
            try:
                # Get the bot's response
                response = app.bot.chat(question)
                print(f"🤖 Answer: {response['response']}")
                
                # Show which sources were used
                if response['retrieved_sources']:
                    print(f"📚 Sources: {', '.join(response['retrieved_sources'])}")
                
            except Exception as e:
                print(f"❌ Error answering question: {str(e)}")
        
        # Clean up
        app.cleanup()
        
    except Exception as e:
        print(f"❌ Demo Error: {str(e)}")

# Main execution logic
if __name__ == "__main__":
    # Check if user wants demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_questions()  # Run quick demo
    else:
        main()  # Run full interactive menu 