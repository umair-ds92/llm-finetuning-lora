#!/usr/bin/env python3
"""
Script to validate dataset quality and generate quality reports.

This script:
1. Loads dataset from file
2. Validates schema and format
3. Checks for quality issues
4. Generates detailed quality report
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from datasets import Dataset

from src.data import get_dataset_statistics, validate_dataset_format
from src.utils.logging import setup_logger

logger = setup_logger(__name__, log_file="logs/dataset_validation.log")


def check_empty_fields(dataset: Dataset) -> Dict:
    """Check for empty or missing fields."""
    logger.info("Checking for empty fields...")
    
    empty_counts = {
        'instruction': 0,
        'input': 0,
        'output': 0,
    }
    
    for example in dataset:
        if not example.get('instruction', '').strip():
            empty_counts['instruction'] += 1
        if not example.get('input', '').strip():
            empty_counts['input'] += 1
        if not example.get('output', '').strip():
            empty_counts['output'] += 1
    
    return empty_counts


def check_length_distribution(dataset: Dataset) -> Dict:
    """Analyze length distribution of text fields."""
    logger.info("Analyzing length distributions...")
    
    instruction_lengths = [len(x['instruction']) for x in dataset]
    input_lengths = [len(x['input']) for x in dataset]
    output_lengths = [len(x['output']) for x in dataset]
    
    def get_percentiles(lengths):
        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        return {
            'p10': sorted_lengths[int(n * 0.1)],
            'p25': sorted_lengths[int(n * 0.25)],
            'p50': sorted_lengths[int(n * 0.5)],
            'p75': sorted_lengths[int(n * 0.75)],
            'p90': sorted_lengths[int(n * 0.9)],
            'p95': sorted_lengths[int(n * 0.95)],
            'p99': sorted_lengths[int(n * 0.99)],
        }
    
    return {
        'instruction': get_percentiles(instruction_lengths),
        'input': get_percentiles(input_lengths),
        'output': get_percentiles(output_lengths),
    }


def check_duplicates(dataset: Dataset) -> Dict:
    """Check for duplicate examples."""
    logger.info("Checking for duplicates...")
    
    # Check instruction+input duplicates
    keys = [
        (x['instruction'], x['input'])
        for x in dataset
    ]
    
    key_counts = Counter(keys)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    
    return {
        'total_duplicates': len(duplicates),
        'duplicate_examples': sum(duplicates.values()) - len(duplicates),
        'duplicate_rate': len(duplicates) / len(dataset) if len(dataset) > 0 else 0,
    }


def check_categories(dataset: Dataset) -> Dict:
    """Analyze category distribution."""
    logger.info("Analyzing category distribution...")
    
    if 'category' not in dataset.column_names:
        return {'note': 'No category field found'}
    
    categories = Counter(dataset['category'])
    
    return {
        'num_categories': len(categories),
        'categories': dict(categories.most_common()),
        'min_count': min(categories.values()) if categories else 0,
        'max_count': max(categories.values()) if categories else 0,
    }


def check_quality_issues(dataset: Dataset) -> List[Dict]:
    """Identify potential quality issues."""
    logger.info("Identifying quality issues...")
    
    issues = []
    
    for idx, example in enumerate(dataset):
        example_issues = []
        
        # Check for very short texts
        if len(example['instruction']) < 10:
            example_issues.append("Very short instruction (<10 chars)")
        if len(example['input']) < 20:
            example_issues.append("Very short input (<20 chars)")
        if len(example['output']) < 20:
            example_issues.append("Very short output (<20 chars)")
        
        # Check for very long texts
        if len(example['instruction']) > 1000:
            example_issues.append("Very long instruction (>1000 chars)")
        if len(example['input']) > 5000:
            example_issues.append("Very long input (>5000 chars)")
        if len(example['output']) > 5000:
            example_issues.append("Very long output (>5000 chars)")
        
        # Check for repetitive text
        words = example['output'].lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common_word, count = word_counts.most_common(1)[0]
            if count > len(words) * 0.3:  # More than 30% same word
                example_issues.append(f"Repetitive output ('{most_common_word}' appears {count} times)")
        
        if example_issues:
            issues.append({
                'index': idx,
                'issues': example_issues,
                'preview': example['instruction'][:100] + "...",
            })
    
    return issues


def generate_validation_report(
    dataset: Dataset,
    output_path: Path,
) -> None:
    """Generate comprehensive validation report."""
    logger.info("Generating validation report...")
    
    report = {
        'dataset_info': {
            'num_examples': len(dataset),
            'columns': dataset.column_names,
        },
        'statistics': get_dataset_statistics(dataset),
        'empty_fields': check_empty_fields(dataset),
        'length_distribution': check_length_distribution(dataset),
        'duplicates': check_duplicates(dataset),
        'categories': check_categories(dataset),
        'quality_issues': check_quality_issues(dataset)[:50],  # First 50 issues
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET VALIDATION REPORT")
    print("="*80)
    
    print(f"\nDataset Info:")
    print(f"  Total examples: {report['dataset_info']['num_examples']}")
    print(f"  Columns: {', '.join(report['dataset_info']['columns'])}")
    
    print(f"\nEmpty Fields:")
    for field, count in report['empty_fields'].items():
        percentage = (count / len(dataset)) * 100 if len(dataset) > 0 else 0
        print(f"  {field}: {count} ({percentage:.2f}%)")
    
    print(f"\nDuplicates:")
    print(f"  Unique duplicate keys: {report['duplicates']['total_duplicates']}")
    print(f"  Total duplicate examples: {report['duplicates']['duplicate_examples']}")
    print(f"  Duplicate rate: {report['duplicates']['duplicate_rate']:.2%}")
    
    print(f"\nLength Distribution (characters):")
    for field in ['instruction', 'input', 'output']:
        dist = report['length_distribution'][field]
        print(f"  {field}:")
        print(f"    p50 (median): {dist['p50']}")
        print(f"    p95: {dist['p95']}")
    
    if 'num_categories' in report['categories']:
        print(f"\nCategories:")
        print(f"  Number of categories: {report['categories']['num_categories']}")
        print(f"  Category distribution:")
        for cat, count in list(report['categories']['categories'].items())[:10]:
            print(f"    {cat}: {count}")
    
    print(f"\nQuality Issues:")
    print(f"  Examples with issues: {len(report['quality_issues'])}")
    if report['quality_issues']:
        print(f"  First 5 issues:")
        for issue in report['quality_issues'][:5]:
            print(f"    Example {issue['index']}: {', '.join(issue['issues'])}")
    
    print("\n" + "="*80)
    print(f"Full report saved to: {output_path}")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Validate dataset quality and generate reports"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset file (JSONL format)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/validation_report.json",
        help="Path to save validation report",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return 1
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load JSONL file
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    dataset = Dataset.from_list(data)
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Validate format
    try:
        validate_dataset_format(dataset)
        logger.info("✓ Dataset format validation passed")
    except ValueError as e:
        logger.error(f"✗ Dataset format validation failed: {str(e)}")
        return 1
    
    # Generate report
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_validation_report(dataset, output_path)
    
    logger.info("Validation complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())