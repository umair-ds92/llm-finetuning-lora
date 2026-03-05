#!/usr/bin/env python3
"""
Training launcher script - simplified interface for training.
"""

import sys
import subprocess

def main():
    """Launch training with common defaults."""
    
    cmd = [
        "python", "src/training/train_lora.py",
        "--config", "configs/training_config.yaml",
    ]
    
    # Add any additional command line arguments
    cmd.extend(sys.argv[1:])
    
    print("Launching training...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()