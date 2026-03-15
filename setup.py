"""
Setup configuration for LLM LoRA Fine-tuning project.
"""

from setuptools import setup, find_packages

setup(
    name="llm-finetuning-lora",
    version="0.1.0",
    description="LLM fine-tuning with LoRA",
    author="Muhammad Umair",
    author_email="umair.ds92@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.35.0",
        "datasets>=2.14.6",
        "peft>=0.6.0",
        "accelerate>=0.24.1",
        "pandas>=2.1.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.2",
        "jsonlines>=4.0.0",
        "tqdm>=4.66.1",
        "evaluate>=0.4.1",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.1",
        "pyyaml>=6.0.1",
        "wandb>=0.15.12",
        "tensorboard>=2.15.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "deployment": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "boto3>=1.29.0",
        ],
    },
)