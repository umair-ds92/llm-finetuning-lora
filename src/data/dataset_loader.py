"""
Dataset loading and preprocessing utilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Load and prepare datasets for fine-tuning."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize dataset loader.
        
        Args:
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._get_default_template()
        
        logger.info(f"Initialized DatasetLoader with max_length={max_length}")
    
    @staticmethod
    def _get_default_template() -> str:
        """Get default Alpaca-style prompt template."""
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )
    
    def load_from_jsonl(
        self,
        file_path: Union[str, Path],
    ) -> Dataset:
        """
        Load dataset from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Read JSONL file
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        return dataset
    
    def load_from_json(
        self,
        file_path: Union[str, Path],
    ) -> Dataset:
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            data = data.get('data', [])
        
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        
        dataset = Dataset.from_list(data)
        return dataset
    
    def load_from_csv(
        self,
        file_path: Union[str, Path],
    ) -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} examples from {file_path}")
        
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def load_splits(
        self,
        train_path: Union[str, Path],
        validation_path: Union[str, Path],
        test_path: Optional[Union[str, Path]] = None,
    ) -> DatasetDict:
        """
        Load pre-split datasets.
        
        Args:
            train_path: Path to training data
            validation_path: Path to validation data
            test_path: Path to test data (optional)
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        splits = {}
        
        # Load train split
        splits['train'] = self.load_from_jsonl(train_path)
        
        # Load validation split
        splits['validation'] = self.load_from_jsonl(validation_path)
        
        # Load test split if provided
        if test_path:
            splits['test'] = self.load_from_jsonl(test_path)
        
        dataset_dict = DatasetDict(splits)
        
        logger.info(
            f"Loaded dataset splits: "
            f"train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}"
            + (f", test={len(splits['test'])}" if 'test' in splits else "")
        )
        
        return dataset_dict
    
    def format_example(self, example: Dict) -> str:
        """
        Format a single example using the prompt template.
        
        Args:
            example: Dictionary with instruction, input, output
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(
            instruction=example.get('instruction', ''),
            input=example.get('input', ''),
            output=example.get('output', ''),
        )
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize examples for training.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples
        """
        # Format prompts
        prompts = [
            self.format_example({
                'instruction': inst,
                'input': inp,
                'output': out,
            })
            for inst, inp, out in zip(
                examples['instruction'],
                examples['input'],
                examples['output'],
            )
        ]
        
        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors=None,
        )
        
        # Add labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4,
        remove_columns: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Prepare dataset for training by tokenizing.
        
        Args:
            dataset: Input dataset
            num_proc: Number of processes for parallel processing
            remove_columns: Columns to remove after tokenization
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Tokenizing dataset...")
        
        # Default columns to remove
        if remove_columns is None:
            remove_columns = ['instruction', 'input', 'output']
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=remove_columns,
            desc="Tokenizing examples",
        )
        
        logger.info(f"Tokenized {len(tokenized_dataset)} examples")
        
        return tokenized_dataset


def validate_dataset_format(dataset: Dataset) -> bool:
    """
    Validate that dataset has required fields.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ['instruction', 'input', 'output']
    
    for field in required_fields:
        if field not in dataset.column_names:
            raise ValueError(
                f"Dataset missing required field: {field}. "
                f"Found columns: {dataset.column_names}"
            )
    
    logger.info("Dataset format validation passed")
    return True


def get_dataset_statistics(dataset: Dataset) -> Dict:
    """
    Calculate statistics for a dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_examples': len(dataset),
        'column_names': dataset.column_names,
    }
    
    # Calculate text length statistics
    if 'instruction' in dataset.column_names:
        instruction_lengths = [len(x) for x in dataset['instruction']]
        stats['instruction_length'] = {
            'mean': sum(instruction_lengths) / len(instruction_lengths),
            'min': min(instruction_lengths),
            'max': max(instruction_lengths),
        }
    
    if 'input' in dataset.column_names:
        input_lengths = [len(x) for x in dataset['input']]
        stats['input_length'] = {
            'mean': sum(input_lengths) / len(input_lengths),
            'min': min(input_lengths),
            'max': max(input_lengths),
        }
    
    if 'output' in dataset.column_names:
        output_lengths = [len(x) for x in dataset['output']]
        stats['output_length'] = {
            'mean': sum(output_lengths) / len(output_lengths),
            'min': min(output_lengths),
            'max': max(output_lengths),
        }
    
    return stats