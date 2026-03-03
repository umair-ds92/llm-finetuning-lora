"""
Unit tests for data processing modules.
"""

import pytest
from datasets import Dataset

from src.data import (
    CybersecurityPreprocessor,
    DataPreprocessor,
    DatasetLoader,
    get_dataset_statistics,
    validate_dataset_format,
)


# Sample test data
SAMPLE_EXAMPLES = [
    {
        "instruction": "Analyze this security alert",
        "input": "Network traffic from IP 192.168.1.100 detected",
        "output": "This appears to be a port scan from a suspicious source",
    },
    {
        "instruction": "Classify the threat level",
        "input": "Malware detected with hash a1b2c3d4e5f6",
        "output": "Severity: HIGH. Immediate action required.",
    },
]


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = DataPreprocessor()
        
        # Test whitespace removal
        text = "  Hello   world  "
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "Hello world"
        
        # Test empty string
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   ") == ""
    
    def test_filter_by_length(self):
        """Test length filtering."""
        preprocessor = DataPreprocessor(
            min_instruction_length=5,
            max_instruction_length=100,
            min_input_length=10,
            max_input_length=100,
            min_output_length=10,
            max_output_length=100,
        )
        
        # Valid example
        valid = {
            "instruction": "Test instruction",
            "input": "This is a test input",
            "output": "This is a test output",
        }
        assert preprocessor.filter_by_length(valid)
        
        # Too short instruction
        too_short = {
            "instruction": "Hi",
            "input": "This is a test input",
            "output": "This is a test output",
        }
        assert not preprocessor.filter_by_length(too_short)
        
        # Too long input
        too_long = {
            "instruction": "Test instruction",
            "input": "x" * 150,
            "output": "This is a test output",
        }
        assert not preprocessor.filter_by_length(too_long)
    
    def test_preprocess_example(self):
        """Test example preprocessing."""
        preprocessor = DataPreprocessor()
        
        example = {
            "instruction": "  Test   instruction  ",
            "input": "  Test   input  ",
            "output": "  Test   output  ",
        }
        
        processed = preprocessor.preprocess_example(example)
        
        assert processed["instruction"] == "Test instruction"
        assert processed["input"] == "Test input"
        assert processed["output"] == "Test output"
    
    def test_process_dataset(self):
        """Test full dataset processing."""
        preprocessor = DataPreprocessor(
            min_instruction_length=5,
            min_input_length=10,
            min_output_length=10,
        )
        
        dataset = Dataset.from_list(SAMPLE_EXAMPLES)
        processed = preprocessor.process_dataset(dataset, num_proc=1)
        
        # All examples should pass filters
        assert len(processed) == len(SAMPLE_EXAMPLES)


class TestCybersecurityPreprocessor:
    """Tests for CybersecurityPreprocessor class."""
    
    def test_normalize_cybersecurity_entities(self):
        """Test cybersecurity entity normalization."""
        preprocessor = CybersecurityPreprocessor()
        
        # Test IP normalization
        text = "IP address: 192.168.1.100"
        normalized = preprocessor.normalize_cybersecurity_entities(text)
        assert "192.168.1.100" in normalized
        
        # Test CVE normalization
        text = "Vulnerability cve-2021-1234"
        normalized = preprocessor.normalize_cybersecurity_entities(text)
        assert "CVE-2021-1234" in normalized


class TestDatasetLoader:
    """Tests for DatasetLoader class."""
    
    def test_format_example(self):
        """Test example formatting."""
        from transformers import AutoTokenizer
        
        # Use a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        loader = DatasetLoader(tokenizer=tokenizer, max_length=512)
        
        example = SAMPLE_EXAMPLES[0]
        formatted = loader.format_example(example)
        
        # Check that all fields are included
        assert "Analyze this security alert" in formatted
        assert "Network traffic from IP" in formatted
        assert "port scan" in formatted


class TestDatasetValidation:
    """Tests for dataset validation functions."""
    
    def test_validate_dataset_format_valid(self):
        """Test validation with valid dataset."""
        dataset = Dataset.from_list(SAMPLE_EXAMPLES)
        assert validate_dataset_format(dataset)
    
    def test_validate_dataset_format_invalid(self):
        """Test validation with invalid dataset."""
        invalid_data = [
            {
                "instruction": "Test",
                # Missing 'input' and 'output'
            }
        ]
        dataset = Dataset.from_list(invalid_data)
        
        with pytest.raises(ValueError):
            validate_dataset_format(dataset)
    
    def test_get_dataset_statistics(self):
        """Test statistics generation."""
        dataset = Dataset.from_list(SAMPLE_EXAMPLES)
        stats = get_dataset_statistics(dataset)
        
        assert stats["num_examples"] == 2
        assert "instruction_length" in stats
        assert "input_length" in stats
        assert "output_length" in stats
        
        # Check mean lengths are reasonable
        assert stats["instruction_length"]["mean"] > 0
        assert stats["input_length"]["mean"] > 0
        assert stats["output_length"]["mean"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])