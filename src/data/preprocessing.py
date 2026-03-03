"""
Data preprocessing and quality filtering utilities.
"""

import re
from typing import Any, Dict, List, Optional

from datasets import Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess and clean dataset examples."""
    
    def __init__(
        self,
        min_instruction_length: int = 10,
        max_instruction_length: int = 500,
        min_input_length: int = 20,
        max_input_length: int = 2000,
        min_output_length: int = 20,
        max_output_length: int = 1000,
        remove_duplicates: bool = True,
    ):
        """
        Initialize data preprocessor.
        
        Args:
            min_instruction_length: Minimum instruction length in characters
            max_instruction_length: Maximum instruction length in characters
            min_input_length: Minimum input length in characters
            max_input_length: Maximum input length in characters
            min_output_length: Minimum output length in characters
            max_output_length: Maximum output length in characters
            remove_duplicates: Whether to remove duplicate examples
        """
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.remove_duplicates = remove_duplicates
        
        logger.info("Initialized DataPreprocessor")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_pii(text: str) -> str:
        """
        Remove personally identifiable information from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with PII removed
        """
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (simple pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # SSN pattern
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Credit card numbers (simple pattern)
        text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CREDIT_CARD]', text)
        
        return text
    
    def filter_by_length(self, example: Dict) -> bool:
        """
        Filter example by text length constraints.
        
        Args:
            example: Dataset example
            
        Returns:
            True if example passes length filters
        """
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # Check instruction length
        if len(instruction) < self.min_instruction_length:
            return False
        if len(instruction) > self.max_instruction_length:
            return False
        
        # Check input length
        if len(input_text) < self.min_input_length:
            return False
        if len(input_text) > self.max_input_length:
            return False
        
        # Check output length
        if len(output) < self.min_output_length:
            return False
        if len(output) > self.max_output_length:
            return False
        
        return True
    
    def preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single example.
        
        Args:
            example: Dataset example
            
        Returns:
            Preprocessed example
        """
        # Clean text fields
        if 'instruction' in example:
            example['instruction'] = self.clean_text(example['instruction'])
        
        if 'input' in example:
            example['input'] = self.clean_text(example['input'])
            # Optionally remove PII from input
            # example['input'] = self.remove_pii(example['input'])
        
        if 'output' in example:
            example['output'] = self.clean_text(example['output'])
        
        return example
    
    def process_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4,
    ) -> Dataset:
        """
        Process entire dataset with cleaning and filtering.
        
        Args:
            dataset: Input dataset
            num_proc: Number of processes for parallel processing
            
        Returns:
            Processed dataset
        """
        logger.info(f"Processing dataset with {len(dataset)} examples")
        
        # Preprocess examples
        logger.info("Cleaning text...")
        processed = dataset.map(
            self.preprocess_example,
            num_proc=num_proc,
            desc="Preprocessing examples",
        )
        
        # Filter by length
        logger.info("Filtering by length constraints...")
        initial_count = len(processed)
        processed = processed.filter(
            self.filter_by_length,
            num_proc=num_proc,
            desc="Filtering by length",
        )
        filtered_count = initial_count - len(processed)
        logger.info(f"Filtered out {filtered_count} examples by length")
        
        # Remove duplicates
        if self.remove_duplicates:
            logger.info("Removing duplicates...")
            initial_count = len(processed)
            
            # Create unique key from instruction + input
            def add_unique_key(example):
                example['_unique_key'] = (
                    example['instruction'] + '||' + example['input']
                )
                return example
            
            processed = processed.map(add_unique_key, num_proc=num_proc)
            
            # Remove duplicates based on unique key
            seen = set()
            
            def is_unique(example):
                key = example['_unique_key']
                if key in seen:
                    return False
                seen.add(key)
                return True
            
            processed = processed.filter(is_unique, desc="Removing duplicates")
            
            # Remove temporary unique key
            processed = processed.remove_columns(['_unique_key'])
            
            duplicate_count = initial_count - len(processed)
            logger.info(f"Removed {duplicate_count} duplicate examples")
        
        logger.info(f"Final dataset size: {len(processed)} examples")
        
        return processed


class CybersecurityPreprocessor(DataPreprocessor):
    """Specialized preprocessor for cybersecurity datasets."""
    
    def __init__(self, **kwargs):
        """Initialize cybersecurity preprocessor."""
        super().__init__(**kwargs)
        
        # Cybersecurity-specific patterns to preserve
        self.preserve_patterns = [
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
            r'\b[a-fA-F0-9]{32,64}\b',  # Hashes
            r'\bCVE-\d{4}-\d{4,7}\b',  # CVE IDs
            r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b',  # Domains
        ]
    
    def normalize_cybersecurity_entities(self, text: str) -> str:
        """
        Normalize cybersecurity entities while preserving structure.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized entities
        """
        # Normalize IP addresses (lowercase)
        text = re.sub(
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
            lambda m: m.group(1).lower(),
            text
        )
        
        # Normalize hashes (lowercase)
        text = re.sub(
            r'\b([a-fA-F0-9]{32,64})\b',
            lambda m: m.group(1).lower(),
            text
        )
        
        # Normalize CVE IDs (uppercase)
        text = re.sub(
            r'\b(cve-\d{4}-\d{4,7})\b',
            lambda m: m.group(1).upper(),
            text,
            flags=re.IGNORECASE
        )
        
        return text
    
    def preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess cybersecurity example.
        
        Args:
            example: Dataset example
            
        Returns:
            Preprocessed example
        """
        # Apply base preprocessing
        example = super().preprocess_example(example)
        
        # Apply cybersecurity-specific normalization
        if 'input' in example:
            example['input'] = self.normalize_cybersecurity_entities(
                example['input']
            )
        
        if 'output' in example:
            example['output'] = self.normalize_cybersecurity_entities(
                example['output']
            )
        
        return example