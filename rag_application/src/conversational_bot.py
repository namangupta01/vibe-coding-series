"""
Conversational bot module with memory and RAG capabilities.

This module creates our AI chatbot that can:
1. REMEMBER past conversations (like a human would)
2. RETRIEVE relevant information from documents
3. GENERATE intelligent responses using that information

The bot works like this:
- User asks a question
- Bot searches for relevant documents (RAG retrieval)
- Bot remembers recent conversation (memory)
- Bot generates a response using both retrieved info and memory
- Bot saves this interaction to memory for future reference

Think of this as an AI assistant with a good memory and access to a library.
"""
import logging  # For tracking what happens
from typing import List, Dict, Optional, Tuple  # For type hints
from collections import deque  # For efficient memory management
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # AI models
import torch  # PyTorch for AI/ML
from langchain.schema import Document  # For document objects

from .config import Config  # Our settings
from .vector_store import VectorStore  # For finding relevant documents

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Manages conversation history with a fixed memory size.
    
    This class is like the bot's "short-term memory". It remembers
    recent conversations so the bot can:
    - Refer back to previous questions
    - Maintain context across multiple interactions
    - Give more coherent responses
    
    We limit memory size to prevent the bot from getting overwhelmed
    by too much history.
    """
    
    def __init__(self, memory_size: int = 4):
        """
        Initialize conversation memory.
        
        Args:
            memory_size: Number of conversation turns to remember
                        (4 means it remembers last 4 question-answer pairs)
        """
        self.memory_size = memory_size
        
        # Use deque (double-ended queue) for efficient memory management
        # maxlen=memory_size means it automatically forgets old conversations
        # when we exceed the limit (like a rolling window)
        self.conversations = deque(maxlen=memory_size)
    
    def add_interaction(self, user_input: str, bot_response: str, context: List[Document] = None):
        """
        Add a new interaction to memory.
        
        Every time the user asks something and the bot responds,
        we save this interaction so we can refer to it later.
        
        Args:
            user_input: What the user asked
            bot_response: How the bot responded
            context: Which documents were used to answer (optional)
        """
        # Create a record of this interaction
        interaction = {
            'user_input': user_input,           # The question
            'bot_response': bot_response,       # The answer
            'context': context or [],           # Documents that were used
            'timestamp': torch.Tensor([0]).item()  # Simple timestamp (could be improved)
        }
        
        # Add to memory (automatically removes oldest if memory is full)
        self.conversations.append(interaction)
        
        logger.debug(f"Added interaction to memory. Memory size: {len(self.conversations)}")
    
    def get_conversation_context(self) -> str:
        """
        Get formatted conversation history for context.
        
        This creates a text summary of recent conversations that
        the bot can use to understand what's been discussed.
        
        Returns:
            Formatted conversation history as a string
        """
        if not self.conversations:
            return ""  # No conversation history yet
        
        # Build a formatted string of previous conversations
        context_parts = []
        for i, conv in enumerate(self.conversations):
            context_parts.append(f"Previous conversation {i+1}:")
            context_parts.append(f"User: {conv['user_input']}")
            context_parts.append(f"Assistant: {conv['bot_response']}")
            context_parts.append("")  # Empty line for readability
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear all conversation memory."""
        self.conversations.clear()
        logger.info("Conversation memory cleared")
    
    def get_memory_stats(self) -> Dict:
        """
        Get statistics about current memory state.
        
        Returns:
            Dictionary with memory information
        """
        return {
            'total_interactions': len(self.conversations),        # How many conversations stored
            'memory_size': self.memory_size,                     # Maximum capacity
            'memory_usage': f"{len(self.conversations)}/{self.memory_size}"  # Current usage
        }

class ConversationalBot:
    """
    RAG-powered conversational bot with memory.
    
    This is the main chatbot class that brings everything together:
    - Uses vector store to find relevant documents
    - Uses memory to remember conversations
    - Uses language model to generate responses
    
    It's like having a knowledgeable assistant who can remember
    your conversation and look up information when needed.
    """
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize the conversational bot.
        
        Args:
            vector_store: VectorStore instance for retrieving documents
        """
        self.config = Config()
        self.vector_store = vector_store
        
        # Initialize conversation memory
        self.memory = ConversationMemory(self.config.MEMORY_SIZE)
        
        # Initialize the language model for generating responses
        self._initialize_language_model()
    
    def _initialize_language_model(self):
        """
        Initialize the language model for text generation.
        
        This sets up the AI model that actually generates the bot's responses.
        We use DialoGPT which is specifically designed for conversations.
        """
        logger.info(f"Loading language model: {self.config.LLM_MODEL}")
        
        try:
            # Use DialoGPT which is good for conversations
            model_name = "microsoft/DialoGPT-medium"
            
            # Initialize tokenizer (converts text to numbers and back)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize the actual AI model
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add pad token if it doesn't exist
            # This is needed for proper text processing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline for easy use
            self.generator = pipeline(
                "text-generation",                          # What type of AI task
                model=self.model,                           # Which model to use
                tokenizer=self.tokenizer,                   # How to process text
                device=-1,                                  # Use CPU (change to 0 for GPU)
                pad_token_id=self.tokenizer.eos_token_id,   # Padding settings
                max_length=512,                             # Maximum response length
                do_sample=True,                             # Add some randomness
                temperature=0.7,                            # How creative (0=boring, 1=creative)
                top_p=0.9                                   # Another creativity setting
            )
            
            logger.info("Language model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load language model: {str(e)}")
            # If model loading fails, we'll use simple fallback responses
            self.generator = None
    
    def _format_prompt(self, query: str, context: List[Document], conversation_history: str) -> str:
        """
        Format the input prompt with context and conversation history.
        
        This creates the input that we feed to the AI model. It includes:
        - Previous conversation history
        - Relevant documents from our database
        - The current question
        
        This gives the AI all the information it needs to provide a good answer.
        
        Args:
            query: The user's current question
            context: Documents retrieved from vector store
            conversation_history: Previous conversations
            
        Returns:
            Formatted prompt string for the AI model
        """
        # Format context from retrieved documents
        context_text = ""
        if context:
            context_parts = []
            # Use only the top 3 most relevant documents to avoid overwhelming the AI
            for i, doc in enumerate(context[:3]):
                source = doc.metadata.get('source', 'Unknown')  # Which paper it came from
                content = doc.page_content[:500]                # Limit content length
                context_parts.append(f"Source {i+1} ({source}): {content}")
            context_text = "\n\n".join(context_parts)
        
        # Build the complete prompt
        prompt_parts = []
        
        # Add conversation history if we have any
        if conversation_history:
            prompt_parts.append("Previous conversation context:")
            prompt_parts.append(conversation_history)
            prompt_parts.append("---")  # Separator
        
        # Add relevant documents if we found any
        if context_text:
            prompt_parts.append("Relevant information from documents:")
            prompt_parts.append(context_text)
            prompt_parts.append("---")  # Separator
        
        # Add the current question and ask for an answer
        prompt_parts.append(f"Based on the above information, please answer the following question:")
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _generate_response_simple(self, prompt: str) -> str:
        """
        Generate response using simple rules when AI model is not available.
        
        This is a fallback method that provides basic responses when
        the main AI model fails to load. It uses keyword matching
        to provide relevant information.
        
        Args:
            prompt: Input prompt (we look for keywords in this)
            
        Returns:
            Simple rule-based response
        """
        query_lower = prompt.lower()
        
        # Simple keyword-based responses
        # Check what the user is asking about and provide relevant info
        if "transformer" in query_lower:
            return "The Transformer is a neural network architecture introduced in 'Attention Is All You Need' that relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        elif "attention" in query_lower:
            return "Attention mechanisms allow models to focus on different parts of the input sequence when processing each element, enabling better handling of long-range dependencies."
        elif "bert" in query_lower:
            return "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that uses bidirectional training to understand context from both directions."
        elif "gpt" in query_lower:
            return "GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates text by predicting the next token in a sequence."
        else:
            return "I can help you with questions about transformer architectures, attention mechanisms, BERT, GPT, and other NLP models based on the provided research papers."
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate response using the AI language model.
        
        This is where the AI magic happens! We feed our formatted prompt
        to the language model and get back a generated response.
        
        Args:
            prompt: Formatted input prompt with context and question
            
        Returns:
            AI-generated response
        """
        # If AI model isn't available, use simple fallback
        if self.generator is None:
            return self._generate_response_simple(prompt)
        
        try:
            # Limit prompt length to avoid model limitations
            max_prompt_length = 400
            if len(prompt) > max_prompt_length:
                prompt = prompt[-max_prompt_length:]  # Use the last part (most relevant)
            
            # Generate response using the AI model
            response = self.generator(
                prompt,                           # Input text
                max_new_tokens=150,              # Maximum length of generated response
                num_return_sequences=1,          # Generate one response
                pad_token_id=self.tokenizer.eos_token_id,  # Padding settings
                eos_token_id=self.tokenizer.eos_token_id   # End-of-sequence settings
            )
            
            # Extract the generated text from the response
            generated_text = response[0]['generated_text']
            
            # Remove the original prompt from the response
            # (The model returns prompt + generated text, we only want the new part)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Clean up the response
            generated_text = generated_text.split('\n')[0]  # Take only the first line
            generated_text = generated_text.strip()         # Remove extra whitespace
            
            # If we got an empty response, use fallback
            if not generated_text:
                return self._generate_response_simple(prompt)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # If anything goes wrong, use the simple fallback
            return self._generate_response_simple(prompt)
    
    def chat(self, user_input: str) -> Dict[str, any]:
        """
        Process user input and generate a response.
        
        This is the main function that handles a complete conversation turn:
        1. Retrieve relevant documents
        2. Get conversation history
        3. Format everything into a prompt
        4. Generate response
        5. Save interaction to memory
        6. Return result with metadata
        
        Args:
            user_input: What the user asked
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing user input: {user_input[:50]}...")
        
        # Step 1: Retrieve relevant context from documents
        context = []
        if self.vector_store and self.vector_store.vector_store:
            try:
                # Search for the 3 most relevant document chunks
                context = self.vector_store.similarity_search(user_input, k=3)
                logger.debug(f"Retrieved {len(context)} context documents")
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
        
        # Step 2: Get conversation history for context
        conversation_history = self.memory.get_conversation_context()
        
        # Step 3: Format everything into a prompt for the AI
        prompt = self._format_prompt(user_input, context, conversation_history)
        
        # Step 4: Generate response using AI model
        response = self._generate_response(prompt)
        
        # Step 5: Save this interaction to memory for future reference
        self.memory.add_interaction(user_input, response, context)
        
        # Step 6: Prepare comprehensive result
        result = {
            'user_input': user_input,                           # What was asked
            'response': response,                               # Bot's answer
            'context_documents': len(context),                  # How many docs were used
            'memory_stats': self.memory.get_memory_stats(),     # Memory usage info
            'retrieved_sources': [doc.metadata.get('source', 'Unknown') for doc in context]  # Which papers were referenced
        }
        
        logger.info("Response generated successfully")
        return result
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get the full conversation history.
        
        Returns:
            List of all conversation interactions stored in memory
        """
        return list(self.memory.conversations)
    
    def clear_conversation_history(self):
        """Clear all conversation history."""
        self.memory.clear_memory()
        logger.info("Conversation history cleared")
    
    def set_vector_store(self, vector_store: VectorStore):
        """
        Set or update the vector store for document retrieval.
        
        This allows us to change which documents the bot has access to.
        
        Args:
            vector_store: New VectorStore instance to use
        """
        self.vector_store = vector_store
        logger.info("Vector store updated for conversational bot") 