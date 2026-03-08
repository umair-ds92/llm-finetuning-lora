#!/usr/bin/env python3
"""
Evaluation launcher script - simplified interface for evaluation.
"""

import sys
import subprocess

def main():
    """Launch evaluation with common defaults."""
    
    cmd = [
        "python", "src/evaluation/evaluate.py",
        "--base-model", "gpt2",
        "--adapter-path", "outputs/test-run",
        "--test-data", "data/splits/test.jsonl",
        "--output-dir", "outputs/evaluation",
        "--compare-with-base",
    ]
    
    # Add any additional command line arguments
    cmd.extend(sys.argv[1:])
    
    print("Launching evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()