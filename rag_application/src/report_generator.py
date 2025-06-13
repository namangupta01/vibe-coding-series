"""
Report generator for creating comprehensive PDF reports of RAG application performance.
"""
import logging
from typing import Dict, List, Any, Optional
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np
import pandas as pd

from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGReportGenerator:
    """Generates comprehensive PDF reports for RAG application evaluation."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.config = Config()
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkred
        )
    
    def generate_report(self, 
                       evaluation_results: Dict[str, Any], 
                       system_info: Dict[str, Any],
                       filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive PDF report.
        
        Args:
            evaluation_results: Results from RAG evaluation
            system_info: System configuration and statistics
            filename: Optional custom filename
            
        Returns:
            Path to generated PDF report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"RAG_Application_Report_{timestamp}.pdf"
        
        filepath = os.path.join(self.config.REPORTS_DIR, filename)
        os.makedirs(self.config.REPORTS_DIR, exist_ok=True)
        
        logger.info(f"Generating PDF report: {filename}")
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=1*inch)
        story = []
        
        # Build report content
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        story.extend(self._create_overview_section(system_info))
        story.append(PageBreak())
        
        story.extend(self._create_technical_implementation_section(system_info))
        story.append(PageBreak())
        
        story.extend(self._create_evaluation_section(evaluation_results))
        story.append(PageBreak())
        
        story.extend(self._create_results_analysis_section(evaluation_results))
        story.append(PageBreak())
        
        story.extend(self._create_test_questions_section(evaluation_results))
        story.append(PageBreak())
        
        story.extend(self._create_conclusion_section(evaluation_results))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated successfully: {filepath}")
        return filepath
    
    def _create_title_page(self) -> List:
        """Create the title page of the report."""
        content = []
        
        # Main title
        content.append(Paragraph("RAG Application", self.title_style))
        content.append(Spacer(1, 20))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=18,
            alignment=1,
            textColor=colors.darkblue
        )
        content.append(Paragraph("Retrieval-Augmented Generation System", subtitle_style))
        content.append(Paragraph("Implementation and Evaluation Report", subtitle_style))
        content.append(Spacer(1, 40))
        
        # Report details
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=1,
            spaceAfter=10
        )
        
        current_date = datetime.now().strftime("%B %d, %Y")
        content.append(Paragraph(f"Report Generated: {current_date}", info_style))
        content.append(Spacer(1, 20))
        
        # Assignment details
        assignment_style = ParagraphStyle(
            'AssignmentStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=1,
            textColor=colors.gray
        )
        
        assignment_text = """
        <b>Assignment:</b> Develop a RAG Application<br/>
        <b>Objective:</b> Design and implement a RAG application that ingests content from 5 PDF documents,
        creates a vector database for semantic retrieval, and powers a conversational bot with
        memory for the last 4 interactions.
        """
        content.append(Paragraph(assignment_text, assignment_style))
        
        return content
    
    def _create_overview_section(self, system_info: Dict[str, Any]) -> List:
        """Create the overview section."""
        content = []
        
        content.append(Paragraph("1. Overview", self.heading_style))
        
        overview_text = """
        This report presents the implementation and evaluation of a Retrieval-Augmented Generation (RAG) 
        application designed to answer questions about transformer-based language models using content 
        from five research papers. The system combines semantic document retrieval with conversational 
        AI to provide accurate, contextually-aware responses.
        """
        content.append(Paragraph(overview_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # System Architecture
        content.append(Paragraph("1.1 System Architecture", self.subheading_style))
        
        architecture_text = """
        The RAG application consists of four main components:
        <br/>• <b>PDF Processing Engine:</b> Downloads and extracts text from research papers
        <br/>• <b>Vector Database:</b> Creates and manages embeddings for semantic search
        <br/>• <b>Conversational Bot:</b> Generates responses using retrieved context and conversation memory
        <br/>• <b>Evaluation Framework:</b> Assesses system performance using RAGAS and custom metrics
        """
        content.append(Paragraph(architecture_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Key Features
        content.append(Paragraph("1.2 Key Features", self.subheading_style))
        
        features_text = """
        <br/>• Semantic retrieval using sentence transformers
        <br/>• Conversational memory for context-aware responses
        <br/>• Multi-source document processing
        <br/>• Comprehensive evaluation using established metrics
        <br/>• Interactive chat interface
        """
        content.append(Paragraph(features_text, self.styles['Normal']))
        
        return content
    
    def _create_technical_implementation_section(self, system_info: Dict[str, Any]) -> List:
        """Create the technical implementation section."""
        content = []
        
        content.append(Paragraph("2. Technical Implementation", self.heading_style))
        
        # Configuration details
        config = system_info.get('config', {})
        
        # Create configuration table
        config_data = [
            ['Component', 'Configuration', 'Value'],
            ['PDF Sources', 'Number of Documents', str(config.get('pdf_count', 5))],
            ['Text Processing', 'Chunk Size', f"{config.get('chunk_size', 1000)} characters"],
            ['Text Processing', 'Chunk Overlap', f"{config.get('chunk_overlap', 200)} characters"],
            ['Embeddings', 'Model', config.get('embedding_model', 'N/A')],
            ['Language Model', 'Model', config.get('llm_model', 'N/A')],
            ['Memory', 'Conversation History', f"{config.get('memory_size', 4)} interactions"],
        ]
        
        config_table = Table(config_data, colWidths=[2*inch, 2*inch, 2*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(Paragraph("2.1 System Configuration", self.subheading_style))
        content.append(config_table)
        content.append(Spacer(1, 20))
        
        # Implementation details
        content.append(Paragraph("2.2 Implementation Details", self.subheading_style))
        
        impl_text = """
        <b>PDF Processing:</b> The system uses PyPDF2 for text extraction with post-processing 
        to handle common PDF artifacts. Text is segmented using LangChain's RecursiveCharacterTextSplitter 
        for optimal chunk sizes.<br/><br/>
        
        <b>Vector Database:</b> FAISS (Facebook AI Similarity Search) provides efficient similarity 
        search capabilities. Documents are embedded using sentence-transformers for semantic understanding.<br/><br/>
        
        <b>Conversational AI:</b> The bot uses Microsoft's DialoGPT model for response generation, 
        enhanced with retrieved context and conversation memory management.<br/><br/>
        
        <b>Evaluation:</b> Performance is assessed using RAGAS metrics (faithfulness, answer relevancy, 
        context precision, context recall) and custom metrics for response quality.
        """
        content.append(Paragraph(impl_text, self.styles['Normal']))
        
        return content
    
    def _create_evaluation_section(self, evaluation_results: Dict[str, Any]) -> List:
        """Create the evaluation methodology section."""
        content = []
        
        content.append(Paragraph("3. Evaluation Methodology", self.heading_style))
        
        methodology_text = """
        The RAG application was evaluated using a comprehensive framework combining established 
        metrics (RAGAS) with custom performance indicators to assess multiple dimensions of system performance.
        """
        content.append(Paragraph(methodology_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Test Questions
        content.append(Paragraph("3.1 Test Questions", self.subheading_style))
        
        questions_text = """
        Ten carefully crafted questions were designed to test various aspects of the system:
        <br/>• Factual knowledge retrieval
        <br/>• Conceptual understanding
        <br/>• Comparative analysis
        <br/>• Technical explanation capabilities
        <br/>• Cross-document reasoning
        """
        content.append(Paragraph(questions_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Evaluation Metrics
        content.append(Paragraph("3.2 Evaluation Metrics", self.subheading_style))
        
        metrics_text = """
        <b>RAGAS Metrics:</b><br/>
        • <b>Faithfulness:</b> Measures factual consistency with source documents<br/>
        • <b>Answer Relevancy:</b> Assesses relevance of responses to questions<br/>
        • <b>Context Precision:</b> Evaluates precision of retrieved context<br/>
        • <b>Context Recall:</b> Measures recall of relevant context<br/><br/>
        
        <b>Custom Metrics:</b><br/>
        • <b>Response Length:</b> Optimal response length scoring<br/>
        • <b>Context Usage:</b> How well responses utilize retrieved context<br/>
        • <b>Coherence:</b> Linguistic and logical coherence assessment<br/>
        • <b>Response Time:</b> System performance and efficiency
        """
        content.append(Paragraph(metrics_text, self.styles['Normal']))
        
        return content
    
    def _create_results_analysis_section(self, evaluation_results: Dict[str, Any]) -> List:
        """Create the results and analysis section."""
        content = []
        
        content.append(Paragraph("4. Results and Analysis", self.heading_style))
        
        # Extract results
        ragas_scores = evaluation_results.get('ragas_scores', {})
        custom_scores = evaluation_results.get('custom_scores', {})
        summary = evaluation_results.get('summary', {})
        
        # Overall Performance
        content.append(Paragraph("4.1 Overall Performance", self.subheading_style))
        
        overall_score = summary.get('overall_score', 0)
        performance_text = f"""
        The RAG application achieved an overall score of <b>{overall_score:.3f}</b> out of 1.000, 
        indicating {'excellent' if overall_score > 0.8 else 'good' if overall_score > 0.6 else 'moderate'} 
        performance across all evaluation criteria.
        """
        content.append(Paragraph(performance_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # RAGAS Results Table
        content.append(Paragraph("4.2 RAGAS Evaluation Results", self.subheading_style))
        
        ragas_data = [['Metric', 'Score', 'Interpretation']]
        for metric, score in ragas_scores.items():
            interpretation = self._interpret_score(score)
            ragas_data.append([metric.replace('_', ' ').title(), f"{score:.3f}", interpretation])
        
        ragas_table = Table(ragas_data, colWidths=[2*inch, 1*inch, 3*inch])
        ragas_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(ragas_table)
        content.append(Spacer(1, 20))
        
        # Custom Metrics Results
        content.append(Paragraph("4.3 Custom Metrics Results", self.subheading_style))
        
        custom_data = [['Metric', 'Score', 'Interpretation']]
        for metric, score in custom_scores.items():
            interpretation = self._interpret_score(score)
            custom_data.append([metric.replace('_', ' ').title(), f"{score:.3f}", interpretation])
        
        custom_table = Table(custom_data, colWidths=[2*inch, 1*inch, 3*inch])
        custom_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(custom_table)
        content.append(Spacer(1, 20))
        
        # Performance Statistics
        content.append(Paragraph("4.4 Performance Statistics", self.subheading_style))
        
        stats_text = f"""
        <b>Response Efficiency:</b><br/>
        • Average Response Time: {summary.get('avg_response_time', 0):.3f} seconds<br/>
        • Average Context Documents: {summary.get('avg_context_docs', 0):.1f}<br/>
        • Total Questions Processed: {summary.get('total_questions', 0)}<br/><br/>
        
        <b>System Performance:</b><br/>
        The system demonstrates efficient retrieval and generation capabilities with consistent 
        response times and appropriate context utilization.
        """
        content.append(Paragraph(stats_text, self.styles['Normal']))
        
        return content
    
    def _create_test_questions_section(self, evaluation_results: Dict[str, Any]) -> List:
        """Create the test questions and answers section."""
        content = []
        
        content.append(Paragraph("5. Test Questions and Responses", self.heading_style))
        
        test_results = evaluation_results.get('test_results', [])
        
        for i, result in enumerate(test_results[:5], 1):  # Show first 5 questions
            content.append(Paragraph(f"5.{i} Question {i}", self.subheading_style))
            
            # Question
            question_style = ParagraphStyle(
                'QuestionStyle',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            content.append(Paragraph(f"Q: {result['question']}", question_style))
            content.append(Spacer(1, 10))
            
            # Answer
            answer_style = ParagraphStyle(
                'AnswerStyle',
                parent=self.styles['Normal'],
                fontSize=10,
                leftIndent=20
            )
            content.append(Paragraph(f"A: {result['answer']}", answer_style))
            content.append(Spacer(1, 10))
            
            # Sources
            if result.get('retrieved_sources'):
                sources_text = f"Sources: {', '.join(result['retrieved_sources'])}"
                source_style = ParagraphStyle(
                    'SourceStyle',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    textColor=colors.gray,
                    leftIndent=20
                )
                content.append(Paragraph(sources_text, source_style))
            
            content.append(Spacer(1, 20))
        
        # Note about remaining questions
        if len(test_results) > 5:
            note_text = f"<i>Note: Showing first 5 questions. Complete results for all {len(test_results)} questions are available in the evaluation data.</i>"
            content.append(Paragraph(note_text, self.styles['Normal']))
        
        return content
    
    def _create_conclusion_section(self, evaluation_results: Dict[str, Any]) -> List:
        """Create the conclusion section."""
        content = []
        
        content.append(Paragraph("6. Conclusion", self.heading_style))
        
        summary = evaluation_results.get('summary', {})
        overall_score = summary.get('overall_score', 0)
        
        # Summary
        content.append(Paragraph("6.1 Summary of Findings", self.subheading_style))
        
        summary_text = f"""
        The RAG application successfully demonstrates the integration of document retrieval, 
        semantic search, and conversational AI. With an overall score of {overall_score:.3f}, 
        the system shows {self._get_performance_level(overall_score)} performance in answering 
        questions about transformer-based language models.
        """
        content.append(Paragraph(summary_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Strengths
        content.append(Paragraph("6.2 System Strengths", self.subheading_style))
        
        strengths_text = """
        <br/>• Effective semantic retrieval using state-of-the-art embeddings
        <br/>• Robust conversation memory management
        <br/>• Comprehensive evaluation framework
        <br/>• Scalable architecture for additional documents
        <br/>• Clear separation of concerns in system design
        """
        content.append(Paragraph(strengths_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Areas for Improvement
        content.append(Paragraph("6.3 Areas for Improvement", self.subheading_style))
        
        improvements_text = """
        <br/>• Enhanced language model fine-tuning for domain-specific responses
        <br/>• Advanced context ranking and filtering mechanisms
        <br/>• Integration of real-time feedback for continuous improvement
        <br/>• Multi-modal support for figures and tables in PDFs
        <br/>• Extended evaluation with human judgments
        """
        content.append(Paragraph(improvements_text, self.styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Future Work
        content.append(Paragraph("6.4 Future Work", self.subheading_style))
        
        future_text = """
        Future enhancements could include integration with larger language models, 
        implementation of the bonus real-time feedback system, and expansion to 
        support multiple document formats and domains. The modular architecture 
        provides a solid foundation for these improvements.
        """
        content.append(Paragraph(future_text, self.styles['Normal']))
        
        return content
    
    def _interpret_score(self, score: float) -> str:
        """Interpret a numerical score into a qualitative assessment."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level description."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "below average" 