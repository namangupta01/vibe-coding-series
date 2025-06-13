#!/usr/bin/env python3
"""
Test script to verify all modules import correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test importing all modules."""
    try:
        print("🧪 Testing module imports...")
        
        from src.config import Config
        print("✅ Config module imported successfully")
        
        from src.pdf_processor import PDFProcessor
        print("✅ PDF Processor module imported successfully")
        
        from src.vector_store import VectorStore
        print("✅ Vector Store module imported successfully")
        
        from src.conversational_bot import ConversationalBot
        print("✅ Conversational Bot module imported successfully")
        
        from src.evaluator import RAGEvaluator
        print("✅ Evaluator module imported successfully")
        
        from src.rag_application import RAGApplication
        print("✅ RAG Application module imported successfully")
        
        from src.report_generator import RAGReportGenerator
        print("✅ Report Generator module imported successfully")
        
        print("\n🎉 All modules imported successfully!")
        
        # Test basic instantiation
        print("\n🔧 Testing basic instantiation...")
        config = Config()
        print(f"✅ Config instantiated - {len(config.PDF_URLS)} PDFs configured")
        
        processor = PDFProcessor()
        print("✅ PDF Processor instantiated")
        
        vector_store = VectorStore()
        print("✅ Vector Store instantiated")
        
        print("\n✨ Basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import/instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 