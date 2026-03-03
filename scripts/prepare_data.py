#!/usr/bin/env python3
"""
Script to prepare and process cybersecurity threat dataset.

This script:
1. Loads raw cybersecurity data
2. Applies quality filtering
3. Formats in instruction-following format
4. Creates train/validation/test splits
5. Generates dataset statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split

from src.data import CybersecurityPreprocessor, get_dataset_statistics
from src.utils.logging import setup_logger

logger = setup_logger(__name__, log_file="logs/data_preparation.log")


def generate_sample_cybersecurity_data(num_examples: int = 100) -> List[Dict]:
    """
    Generate sample cybersecurity threat data for demonstration.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        List of example dictionaries
    """
    logger.info(f"Generating {num_examples} sample examples")
    
    # Sample templates for cybersecurity threats
    threat_types = [
        "Malware Analysis",
        "Phishing Detection",
        "Network Intrusion",
        "Vulnerability Assessment",
        "Incident Response",
        "Threat Intelligence"
    ]
    
    severity_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]
    
    examples = []
    
    for i in range(num_examples):
        threat_type = threat_types[i % len(threat_types)]
        severity = severity_levels[i % len(severity_levels)]
        
        # Generate realistic-looking data
        ip = f"192.168.{(i % 255)}.{((i * 7) % 255)}"
        port = 1000 + (i % 8999)
        hash_val = f"a1b2c3d4e5f6{i:04d}" + "0" * 40
        
        example = {
            "instruction": f"Analyze the following {threat_type.lower()} alert and provide a detailed assessment.",
            "input": (
                f"Alert Type: {threat_type}\n"
                f"Source IP: {ip}\n"
                f"Destination Port: {port}\n"
                f"Timestamp: 2025-02-{(i % 28) + 1:02d} {(i % 24):02d}:{(i % 60):02d}:00 UTC\n"
                f"File Hash: {hash_val[:32]}\n"
                f"Detection Signature: THREAT_{i:04d}\n"
                f"Payload Sample: Suspicious activity detected on network segment..."
            ),
            "output": (
                f"## Threat Assessment\n\n"
                f"**Severity:** {severity}\n\n"
                f"**Classification:** {threat_type}\n\n"
                f"**Analysis:**\n"
                f"The observed activity from IP {ip} targeting port {port} exhibits characteristics "
                f"consistent with {threat_type.lower()}. The file hash {hash_val[:32]} does not match "
                f"known malware signatures but displays suspicious behavior patterns.\n\n"
                f"**Recommended Actions:**\n"
                f"1. Block source IP {ip} at the network perimeter\n"
                f"2. Conduct deeper forensic analysis on affected systems\n"
                f"3. Review security logs for similar patterns\n"
                f"4. Update detection signatures with THREAT_{i:04d}\n\n"
                f"**Priority:** {'Immediate' if severity in ['CRITICAL', 'HIGH'] else 'Standard'} response required."
            ),
            "category": threat_type.lower().replace(" ", "_"),
            "severity": severity,
        }
        
        examples.append(example)
    
    return examples


def load_or_generate_dataset(
    data_path: Path,
    num_examples: int = 10000,
) -> List[Dict]:
    """
    Load dataset from file or generate sample data.
    
    Args:
        data_path: Path to dataset file
        num_examples: Number of examples if generating
        
    Returns:
        List of examples
    """
    if data_path.exists():
        logger.info(f"Loading dataset from {data_path}")
        
        if data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = data.get('data', [])
        elif data_path.suffix == '.jsonl':
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loaded {len(data)} examples from file")
        return data
    else:
        logger.warning(
            f"Dataset file not found: {data_path}. "
            f"Generating {num_examples} sample examples."
        )
        return generate_sample_cybersecurity_data(num_examples)


def create_splits(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Create train/validation/test splits.
    
    Args:
        data: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    logger.info(
        f"Creating splits: train={train_ratio}, "
        f"val={val_ratio}, test={test_ratio}"
    )
    
    # First split: separate test set
    train_val, test = train_test_split(
        data,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True,
    )
    
    splits = {
        'train': train,
        'validation': val,
        'test': test,
    }
    
    logger.info(
        f"Created splits: train={len(train)}, "
        f"val={len(val)}, test={len(test)}"
    )
    
    return splits


def save_splits(
    splits: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Save splits to JSONL files.
    
    Args:
        splits: Dictionary of splits
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        
        logger.info(f"Saving {split_name} split to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in split_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(split_data)} examples to {output_file}")


def save_statistics(
    splits: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Calculate and save dataset statistics.
    
    Args:
        splits: Dictionary of splits
        output_dir: Output directory
    """
    logger.info("Calculating dataset statistics")
    
    stats = {}
    
    for split_name, split_data in splits.items():
        dataset = Dataset.from_list(split_data)
        stats[split_name] = get_dataset_statistics(dataset)
    
    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Statistics Summary")
    print("="*60)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Examples: {split_stats['num_examples']}")
        if 'instruction_length' in split_stats:
            print(f"  Instruction length: {split_stats['instruction_length']['mean']:.1f} chars")
        if 'input_length' in split_stats:
            print(f"  Input length: {split_stats['input_length']['mean']:.1f} chars")
        if 'output_length' in split_stats:
            print(f"  Output length: {split_stats['output_length']['mean']:.1f} chars")
    print("="*60 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare cybersecurity threat dataset for fine-tuning"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/cybersecurity/threats.json",
        help="Path to raw dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Output directory for processed splits",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of examples to generate if data file doesn't exist",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Convert paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    
    # Load configuration
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = {}
        logger.warning(f"Configuration file not found: {config_path}")
    
    # Load or generate dataset
    data = load_or_generate_dataset(data_path, args.num_examples)
    
    # Initialize preprocessor
    preprocessor_config = config.get('quality_filters', {})
    preprocessor = CybersecurityPreprocessor(**preprocessor_config)
    
    # Convert to dataset and preprocess
    dataset = Dataset.from_list(data)
    dataset = preprocessor.process_dataset(dataset)
    
    # Convert back to list
    processed_data = [example for example in dataset]
    
    # Create splits
    split_config = config.get('splits', {})
    splits = create_splits(
        processed_data,
        train_ratio=split_config.get('train', 0.8),
        val_ratio=split_config.get('validation', 0.1),
        test_ratio=split_config.get('test', 0.1),
        seed=args.seed,
    )
    
    # Save splits
    save_splits(splits, output_dir)
    
    # Save statistics
    save_statistics(splits, output_dir)
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()