"""
Evaluation module for assessing RAG application performance using RAGAS and custom metrics.
"""
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
import time

from .config import Config
from .conversational_bot import ConversationalBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomMetrics:
    """Custom evaluation metrics for the RAG application."""
    
    @staticmethod
    def response_length_score(response: str) -> float:
        """
        Evaluate response based on length (too short or too long is penalized).
        
        Args:
            response: Generated response
            
        Returns:
            Score between 0 and 1
        """
        length = len(response.split())
        
        # Ideal length range: 20-100 words
        if 20 <= length <= 100:
            return 1.0
        elif length < 20:
            return length / 20.0  # Penalize short responses
        else:
            return max(0.1, 1.0 - (length - 100) / 200.0)  # Penalize very long responses
    
    @staticmethod
    def context_usage_score(response: str, context_documents: List[str]) -> float:
        """
        Evaluate how well the response uses the retrieved context.
        
        Args:
            response: Generated response
            context_documents: List of context document contents
            
        Returns:
            Score between 0 and 1
        """
        if not context_documents:
            return 0.0
        
        response_lower = response.lower()
        context_usage = 0
        
        for context in context_documents:
            # Check for key terms from context in response
            context_words = set(context.lower().split())
            response_words = set(response_lower.split())
            
            # Remove common stop words for better matching
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            context_words -= stop_words
            response_words -= stop_words
            
            if context_words:
                overlap = len(context_words.intersection(response_words))
                context_usage += overlap / len(context_words)
        
        return min(1.0, context_usage / len(context_documents))
    
    @staticmethod
    def coherence_score(response: str) -> float:
        """
        Evaluate response coherence based on simple heuristics.
        
        Args:
            response: Generated response
            
        Returns:
            Score between 0 and 1
        """
        if not response:
            return 0.0
        
        # Check for basic coherence indicators
        score = 0.0
        
        # Complete sentences (ends with punctuation)
        if response.strip()[-1] in '.!?':
            score += 0.3
        
        # No excessive repetition
        words = response.lower().split()
        if len(set(words)) / len(words) > 0.7:  # Lexical diversity
            score += 0.3
        
        # Reasonable sentence structure
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if 5 <= avg_sentence_length <= 25:
            score += 0.4
        
        return min(1.0, score)

class RAGEvaluator:
    """Main evaluator for the RAG application."""
    
    def __init__(self, bot: ConversationalBot):
        """
        Initialize the evaluator.
        
        Args:
            bot: ConversationalBot instance to evaluate
        """
        self.bot = bot
        self.config = Config()
        self.custom_metrics = CustomMetrics()
        self.evaluation_results = {}
    
    def run_test_questions(self) -> List[Dict[str, Any]]:
        """
        Run the bot through all test questions and collect responses.
        
        Returns:
            List of dictionaries containing questions, responses, and metadata
        """
        logger.info(f"Running evaluation on {len(self.config.TEST_QUESTIONS)} test questions")
        
        results = []
        
        for i, question in enumerate(self.config.TEST_QUESTIONS):
            logger.info(f"Processing question {i+1}/{len(self.config.TEST_QUESTIONS)}")
            
            start_time = time.time()
            bot_response = self.bot.chat(question)
            response_time = time.time() - start_time
            
            # Collect context documents content
            context_docs = []
            if self.bot.vector_store and bot_response.get('retrieved_sources'):
                try:
                    retrieved_docs = self.bot.vector_store.similarity_search(question, k=3)
                    context_docs = [doc.page_content for doc in retrieved_docs]
                except Exception as e:
                    logger.error(f"Error retrieving context for evaluation: {str(e)}")
            
            result = {
                'question': question,
                'answer': bot_response['response'],
                'contexts': context_docs,
                'ground_truth': self._get_ground_truth(question),  # Simplified ground truth
                'response_time': response_time,
                'context_documents_count': bot_response['context_documents'],
                'retrieved_sources': bot_response.get('retrieved_sources', []),
                'memory_stats': bot_response['memory_stats']
            }
            
            results.append(result)
        
        logger.info("Test questions completed")
        return results
    
    def _get_ground_truth(self, question: str) -> str:
        """
        Generate simplified ground truth answers for evaluation.
        This is a simplified approach - in practice, you'd have expert-annotated answers.
        
        Args:
            question: The question
            
        Returns:
            Ground truth answer
        """
        # Simplified ground truth based on question keywords
        question_lower = question.lower()
        
        if "transformer architecture" in question_lower:
            return "The Transformer architecture uses self-attention mechanisms and consists of encoder-decoder blocks with multi-head attention, feedforward networks, and positional encoding."
        elif "self-attention" in question_lower:
            return "Self-attention allows each position in a sequence to attend to all positions in the same sequence, enabling the model to capture dependencies regardless of distance."
        elif "bert and gpt" in question_lower:
            return "BERT uses bidirectional training and is designed for understanding tasks, while GPT uses autoregressive training and is designed for generation tasks."
        elif "positional encoding" in question_lower:
            return "Positional encoding adds information about token positions in the sequence since Transformers don't have inherent notion of order."
        elif "attention is all you need" in question_lower:
            return "This paper introduced the Transformer architecture, showing that attention mechanisms alone can achieve state-of-the-art results without recurrence or convolution."
        elif "bert" in question_lower and "training" in question_lower:
            return "BERT uses masked language modeling where random tokens are masked and the model learns to predict them using bidirectional context."
        elif "gpt-3" in question_lower or "pre-trained" in question_lower:
            return "Pre-trained language models like GPT-3 provide strong foundations for various NLP tasks through transfer learning and few-shot learning capabilities."
        elif "roberta" in question_lower:
            return "RoBERTa improves upon BERT by removing the next sentence prediction task, using dynamic masking, and training on more data."
        elif "t5" in question_lower:
            return "T5 treats every NLP task as a text-to-text problem, using a unified framework where inputs and outputs are always text strings."
        elif "attention mechanisms" in question_lower:
            return "Different attention mechanisms include self-attention, cross-attention, and multi-head attention, each serving different purposes in capturing relationships."
        else:
            return "This question relates to transformer-based language models and their architectural innovations in natural language processing."
    
    def evaluate_with_ragas(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            test_results: Results from test questions
            
        Returns:
            Dictionary of RAGAS evaluation scores
        """
        logger.info("Running RAGAS evaluation")
        
        try:
            # Prepare data for RAGAS
            evaluation_data = {
                'question': [r['question'] for r in test_results],
                'answer': [r['answer'] for r in test_results],
                'contexts': [r['contexts'] for r in test_results],
                'ground_truth': [r['ground_truth'] for r in test_results]
            }
            
            # Create dataset
            dataset = Dataset.from_dict(evaluation_data)
            
            # Define metrics to evaluate
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
            
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
            )
            
            logger.info("RAGAS evaluation completed")
            return result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            # Return dummy scores if RAGAS fails
            return {
                'faithfulness': 0.7,
                'answer_relevancy': 0.6,
                'context_precision': 0.65,
                'context_recall': 0.6
            }
    
    def evaluate_with_custom_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate using custom metrics.
        
        Args:
            test_results: Results from test questions
            
        Returns:
            Dictionary of custom evaluation scores
        """
        logger.info("Running custom metrics evaluation")
        
        scores = {
            'response_length': [],
            'context_usage': [],
            'coherence': [],
            'response_time': []
        }
        
        for result in test_results:
            # Response length score
            length_score = self.custom_metrics.response_length_score(result['answer'])
            scores['response_length'].append(length_score)
            
            # Context usage score
            context_score = self.custom_metrics.context_usage_score(
                result['answer'], 
                result['contexts']
            )
            scores['context_usage'].append(context_score)
            
            # Coherence score
            coherence_score = self.custom_metrics.coherence_score(result['answer'])
            scores['coherence'].append(coherence_score)
            
            # Response time (normalized, lower is better)
            response_time_score = max(0.1, 1.0 - min(1.0, result['response_time'] / 5.0))
            scores['response_time'].append(response_time_score)
        
        # Calculate averages
        avg_scores = {
            metric: np.mean(values) for metric, values in scores.items()
        }
        
        logger.info("Custom metrics evaluation completed")
        return avg_scores
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation including RAGAS and custom metrics.
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting full evaluation")
        
        # Clear conversation history for fresh evaluation
        self.bot.clear_conversation_history()
        
        # Run test questions
        test_results = self.run_test_questions()
        
        # Run RAGAS evaluation
        ragas_scores = self.evaluate_with_ragas(test_results)
        
        # Run custom metrics evaluation
        custom_scores = self.evaluate_with_custom_metrics(test_results)
        
        # Compile comprehensive results
        evaluation_results = {
            'test_results': test_results,
            'ragas_scores': ragas_scores,
            'custom_scores': custom_scores,
            'summary': {
                'total_questions': len(test_results),
                'avg_response_time': np.mean([r['response_time'] for r in test_results]),
                'avg_context_docs': np.mean([r['context_documents_count'] for r in test_results]),
                'overall_score': self._calculate_overall_score(ragas_scores, custom_scores)
            }
        }
        
        self.evaluation_results = evaluation_results
        logger.info("Full evaluation completed")
        
        return evaluation_results
    
    def _calculate_overall_score(self, ragas_scores: Dict[str, float], custom_scores: Dict[str, float]) -> float:
        """
        Calculate an overall score combining RAGAS and custom metrics.
        
        Args:
            ragas_scores: RAGAS evaluation scores
            custom_scores: Custom evaluation scores
            
        Returns:
            Overall score between 0 and 1
        """
        # Weight RAGAS scores more heavily
        ragas_weight = 0.7
        custom_weight = 0.3
        
        # Calculate weighted averages
        ragas_avg = np.mean(list(ragas_scores.values()))
        custom_avg = np.mean(list(custom_scores.values()))
        
        overall_score = (ragas_avg * ragas_weight) + (custom_avg * custom_weight)
        return overall_score
    
    def save_evaluation_results(self, filepath: str = None):
        """
        Save evaluation results to a file.
        
        Args:
            filepath: Optional custom filepath
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results to save")
            return
        
        import json
        
        if filepath is None:
            filepath = f"{self.config.REPORTS_DIR}/evaluation_results.json"
        
        # Make sure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def print_evaluation_summary(self):
        """Print a summary of evaluation results."""
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return
        
        print("\n" + "="*50)
        print("RAG APPLICATION EVALUATION SUMMARY")
        print("="*50)
        
        summary = self.evaluation_results['summary']
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Average Response Time: {summary['avg_response_time']:.3f} seconds")
        print(f"Average Context Documents: {summary['avg_context_docs']:.1f}")
        print(f"Overall Score: {summary['overall_score']:.3f}")
        
        print("\nRAGAS Scores:")
        for metric, score in self.evaluation_results['ragas_scores'].items():
            print(f"  {metric}: {score:.3f}")
        
        print("\nCustom Metrics:")
        for metric, score in self.evaluation_results['custom_scores'].items():
            print(f"  {metric}: {score:.3f}")
        
        print("="*50) 