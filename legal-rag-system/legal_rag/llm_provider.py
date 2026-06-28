"""
LLM Provider abstraction layer for Legal RAG System.
Uses Groq API directly with full LangChain Runnable compatibility.
"""

import os
from typing import Optional, Dict, Any, List
from rich.console import Console

# Import Runnable to make GroqLLMWrapper compatible with LangChain pipelines
from langchain_core.runnables import Runnable

console = Console()


class GroqLLMWrapper(Runnable):
    """
    Groq API wrapper that implements LangChain's Runnable interface.
    
    This makes it compatible with the pipe (|) operator in LangChain pipelines.
    The invoke() method accepts various input formats:
    - Plain string
    - ChatPromptValue (from prompt templates)
    - Message objects
    - Dict with messages
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """
        Initialize Groq API wrapper.
        
        Args:
            model_name: Groq model name (e.g., 'llama-3.1-8b-instant')
            api_key: Groq API key
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, input_data, config=None, **kwargs) -> str:
        """
        Invoke Groq API with input data.
        
        This is the main method required by LangChain's Runnable interface.
        It handles all input formats that come through the pipeline.
        
        Args:
            input_data: Input in various formats (string, ChatPromptValue, etc.)
            config: Optional run configuration (ignored)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Generated text response from Groq
        """
        try:
            # Convert any input format to Groq-compatible messages
            messages = self._convert_to_messages(input_data)
            
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            console.print(f"[red]Groq API error: {e}[/red]")
            raise RuntimeError(f"Groq API error: {e}")
    
    def _convert_to_messages(self, input_data) -> List[Dict[str, str]]:
        """
        Convert various input formats to Groq-compatible messages list.
        
        Args:
            input_data: Input in any supported format
            
        Returns:
            List of {"role": ..., "content": ...} dicts
        """
        # Case 1: Plain string (direct query)
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]
        
        # Case 2: Has 'messages' attribute (ChatPromptValue from LangChain)
        if hasattr(input_data, 'messages'):
            return self._extract_messages(input_data.messages)
        
        # Case 3: Dict with 'messages' key
        if isinstance(input_data, dict) and 'messages' in input_data:
            return self._extract_messages(input_data['messages'])
        
        # Case 4: Has 'content' attribute (single message object)
        if hasattr(input_data, 'content'):
            role = self._normalize_role(getattr(input_data, 'type', 'user'))
            return [{"role": role, "content": str(input_data.content)}]
        
        # Case 5: Has 'text' attribute
        if hasattr(input_data, 'text'):
            return [{"role": "user", "content": str(input_data.text)}]
        
        # Case 6: List of messages
        if isinstance(input_data, (list, tuple)):
            return self._extract_messages(input_data)
        
        # Fallback: Convert to string
        return [{"role": "user", "content": str(input_data)}]
    
    def _extract_messages(self, messages) -> List[Dict[str, str]]:
        """
        Extract and normalize messages from various formats.
        
        Args:
            messages: Messages in any format (list of dicts, objects, tuples)
            
        Returns:
            Normalized list of {"role": ..., "content": ...} dicts
        """
        normalized = []
        
        for msg in messages:
            if isinstance(msg, dict):
                # Already a dict
                role = self._normalize_role(msg.get('role', msg.get('type', 'user')))
                content = str(msg.get('content', ''))
                
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain Message object (HumanMessage, AIMessage, SystemMessage)
                role = self._normalize_role(msg.type)
                content = str(msg.content)
                
            elif isinstance(msg, (tuple, list)) and len(msg) >= 2:
                # Tuple format: ("human", "Hello") or ("user", "Hello")
                role = self._normalize_role(str(msg[0]))
                content = str(msg[1])
                
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # Object with role and content attributes
                role = self._normalize_role(msg.role)
                content = str(msg.content)
                
            else:
                # Unknown format - treat as user message
                role = "user"
                content = str(msg)
            
            normalized.append({"role": role, "content": content})
        
        # Ensure at least one message
        if not normalized:
            normalized = [{"role": "user", "content": "Hello"}]
        
        return normalized
    
    def _normalize_role(self, role: str) -> str:
        """
        Normalize role names to Groq-compatible format.
        
        Groq expects: 'system', 'user', 'assistant'
        
        Args:
            role: Role name from any format
            
        Returns:
            Normalized role name
        """
        role = str(role).lower().strip()
        
        # Map common role names
        role_map = {
            'human': 'user',
            'ai': 'assistant',
            'bot': 'assistant',
            'system': 'system',
            'function': 'assistant',
            'tool': 'assistant',
        }
        
        return role_map.get(role, role if role in ('user', 'assistant', 'system') else 'user')


class LLMProvider:
    """
    Unified LLM provider for Legal RAG System.
    
    Supports multiple backends:
    - groq: Fast cloud inference via Groq API (DEFAULT)
    - ollama: Local models via Ollama
    - openai: OpenAI GPT models
    
    All providers return Runnable-compatible LLM instances
    that work with LangChain's pipe (|) operator.
    """
    
    DEFAULT_MODELS = {
        "groq": "llama-3.1-8b-instant",      # Fast, current model
        "ollama": "llama3:8b-instruct",       # Local model
        "openai": "gpt-3.5-turbo"             # OpenAI model
    }
    
    def __init__(
        self,
        provider: str = "groq",
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize LLM provider.
        
        Args:
            provider: LLM provider name ('groq', 'ollama', 'openai')
            model_name: Specific model name (uses default if None)
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        self.model_name = model_name or self.DEFAULT_MODELS.get(self.provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.llm = None  # Will hold the Runnable-compatible LLM
        
        # Validate provider
        if self.provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Choose from: {list(self.DEFAULT_MODELS.keys())}"
            )
        
        # Initialize the LLM
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM backend."""
        console.print(f"\n[bold]🤖 Initializing LLM:[/bold] {self.provider}/{self.model_name}")
        
        init_methods = {
            "groq": self._init_groq,
            "ollama": self._init_ollama,
            "openai": self._init_openai,
        }
        
        init_method = init_methods.get(self.provider)
        if init_method:
            init_method()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        console.print(f"[green]✅ LLM initialized:[/green] {self.provider}/{self.model_name}")
    
    def _init_groq(self):
        """Initialize Groq using Runnable-compatible wrapper."""
        # Get API key from multiple sources
        api_key = (
            self.kwargs.get("api_key") or 
            os.getenv("GROQ_API_KEY") or
            None
        )
        
        if not api_key:
            raise ValueError(
                "\n❌ GROQ_API_KEY is required!\n\n"
                "Set it using one of these methods:\n"
                "  1. PowerShell: $env:GROQ_API_KEY='your-key-here'\n"
                "  2. .env file: GROQ_API_KEY=your-key-here\n"
                "  3. Get a free key at: https://console.groq.com\n"
            )
        
        console.print(f"   🔑 Using Groq API key: ****{api_key[-4:]}")
        console.print(f"   📡 Model: {self.model_name}")
        
        try:
            # Use Runnable-compatible Groq wrapper
            self.llm = GroqLLMWrapper(
                model_name=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            console.print("   ✅ Groq client ready (LangChain Runnable compatible)")
        except ImportError:
            raise ImportError(
                "Groq package not found. Install with:\n"
                "  pip install groq"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq: {e}")
    
    def _init_ollama(self):
        """Initialize Ollama backend."""
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError(
                "Ollama not available. Install with:\n"
                "  pip install ollama langchain-community"
            )
        
        base_url = self.kwargs.get("base_url", "http://localhost:11434")
        
        self.llm = Ollama(
            model=self.model_name,
            base_url=base_url,
            temperature=self.temperature,
        )
    
    def _init_openai(self):
        """Initialize OpenAI backend."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI not available. Install with:\n"
                "  pip install openai langchain-openai"
            )
        
        api_key = self.kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def get_llm(self):
        """
        Get the initialized LLM instance.
        
        Returns:
            Runnable-compatible LLM instance
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized. Check provider configuration.")
        return self.llm
    
    def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with a prompt string.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            LLM response string
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        return self.llm.invoke(prompt)
    
    def test_connection(self) -> bool:
        """
        Test the connection to the LLM provider.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            console.print("   🔍 Testing API connection...")
            response = self.llm.invoke("Say 'OK' if you receive this message.")
            console.print(f"   📩 Response: {response.strip()[:50]}...")
            console.print("[green]   ✅ API connection successful![/green]")
            return True
        except Exception as e:
            console.print(f"[red]   ❌ API connection failed: {e}[/red]")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider."""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "initialized": self.llm is not None
        }