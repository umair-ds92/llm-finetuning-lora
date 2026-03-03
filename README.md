# LLM LoRA Fine-Tuning

Enterprise-grade parameter-efficient fine-tuning pipeline for adapting large language models (Llama-2, Mistral) to domain-specific tasks using LoRA.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 🎯 Overview

Fine-tune large language models efficiently with LoRA for specialized enterprise tasks like cybersecurity threat analysis, legal document processing, and medical diagnostics. This pipeline focuses on **parameter-efficient training** that reduces memory usage by 50% and training time by 75% compared to full fine-tuning.

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/umair-ds92/llm-finetuning-lora.git
cd llm-finetuning-lora

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your HuggingFace token
```

### Prepare Your Dataset
```bash
# Generate sample cybersecurity dataset (100 examples)
python scripts/prepare_data.py --num-examples 100

# Or use your own data
python scripts/prepare_data.py \
    --data-path data/raw/your_data.json \
    --num-examples 10000 \
    --output-dir data/splits

# Validate dataset quality
python scripts/validate_dataset.py \
    --dataset-path data/splits/train.jsonl
```

---

## ✨ Current Features

### Data Processing Pipeline
- **Multi-format support**: Load datasets from JSON, JSONL, or CSV
- **Quality filtering**: Automatic length validation, duplicate removal, PII filtering
- **Domain-specific preprocessing**: Specialized handling for cybersecurity data (IP normalization, hash formatting, CVE standardization)
- **Train/validation/test splitting**: Configurable split ratios with stratification support
- **Comprehensive validation**: Automated quality checks, statistics generation, and detailed reporting

### Model Infrastructure
- **Multi-model support**: Compatible with Llama-2 (7B, 13B) and Mistral (7B) architectures
- **LoRA configuration**: Pre-configured parameter-efficient fine-tuning setup
- **Flexible loading**: Support for full precision, bfloat16, and quantized (4-bit/8-bit) models
- **Tokenizer management**: Automatic tokenizer initialization and configuration

### Development Tools
- **Validation suite**: Quality checks, duplicate detection, length analysis
- **Statistics generation**: Automated dataset profiling and reporting
- **Unit tests**: Comprehensive test coverage for data processing
- **Build automation**: Makefile for common tasks

---

## Project Structure
```
llm-finetuning-lora/
├── configs/                      # Configuration files
│   ├── lora_config.yaml         # LoRA & training settings
│   ├── data_config.yaml         # Data processing config
│   └── deepspeed_config.json    # DeepSpeed optimization
├── src/                          # Source code
│   ├── data/                    # Data processing modules
│   │   ├── dataset_loader.py   # Dataset loading & tokenization
│   │   └── preprocessing.py    # Data cleaning & filtering
│   ├── models/                  # Model management
│   │   └── model_loader.py     # Model loading & LoRA setup
│   └── utils/                   # Utilities
│       └── logging.py          # Logging configuration
├── scripts/                      # Executable scripts
│   ├── prepare_data.py          # Dataset preparation
│   ├── validate_dataset.py      # Data validation
│   └── download_model.py        # Model download utility
├── tests/                        # Unit tests
└── data/                         # Dataset storage
    ├── raw/                     # Raw data files
    ├── processed/               # Cleaned datasets
    └── splits/                  # Train/val/test splits
```

---

## 📊 Dataset Format

Datasets should follow the Alpaca instruction format:
```json
{
  "instruction": "Classify the following security alert",
  "input": "Network traffic anomaly detected from IP 192.168.1.100...",
  "output": "Severity: HIGH. Classification: Port Scanning. Recommended Action: Block IP..."
}
```

**Required fields**: `instruction`, `input`, `output`  
**Optional fields**: `category`, `severity`, `metadata`

---

## ⚙️ Configuration

### Data Processing

Configure dataset processing in `configs/data_config.yaml`:
```yaml
quality_filters:
  min_instruction_length: 10
  max_instruction_length: 500
  min_input_length: 20
  max_input_length: 2000
  min_output_length: 20
  max_output_length: 1000
  remove_duplicates: true

splits:
  train: 0.8
  validation: 0.1
  test: 0.1
```

### LoRA Settings

Configure LoRA parameters in `configs/lora_config.yaml`:
```yaml
lora:
  r: 16                    # Rank (8-64)
  alpha: 32                # Scaling (typically 2x rank)
  dropout: 0.05
  target_modules: [q_proj, v_proj]
```

---

## 🔧 Supported Models

- **Llama-2**: 7B, 7B-Chat, 13B
- **Mistral**: 7B, 7B-Instruct

All models support LoRA/QLoRA fine-tuning with configurable precision (fp16, bfloat16, 4-bit, 8-bit).

---

## 🧪 Testing
```bash
# Run all tests
make test
# or
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

--- 

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

---

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for PEFT and Transformers libraries
- [Meta AI](https://ai.meta.com/) for Llama-2 models
- [Mistral AI](https://mistral.ai/) for Mistral models
- [Microsoft DeepSpeed](https://www.deepspeed.ai/) for optimization frameworks

---