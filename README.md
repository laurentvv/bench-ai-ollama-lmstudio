# Bench AI with Ollama/LMStudio

<div align="center">

[![English](https://img.shields.io/badge/lang-en-blue.svg)](README.md) [![中文](https://img.shields.io/badge/lang-zh-blue.svg)](lang/README.zh.md) [![हिंदी](https://img.shields.io/badge/lang-hi-blue.svg)](lang/README.hi.md) [![Español](https://img.shields.io/badge/lang-es-blue.svg)](lang/README.es.md) [![Français](https://img.shields.io/badge/lang-fr-blue.svg)](lang/README.fr.md) [![العربية](https://img.shields.io/badge/lang-ar-blue.svg)](lang/README.ar.md) [![বাংলা](https://img.shields.io/badge/lang-bn-blue.svg)](lang/README.bn.md) [![Русский](https://img.shields.io/badge/lang-ru-blue.svg)](lang/README.ru.md) [![Português](https://img.shields.io/badge/lang-pt-blue.svg)](lang/README.pt.md) [![Bahasa Indonesia](https://img.shields.io/badge/lang-id-blue.svg)](lang/README.id.md)

![Bench AI Logo](https://img.shields.io/badge/Bench%20AI-LLM%20Benchmark%20Tool-blue)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

Bench AI is a lightweight and efficient tool for evaluating the performance of language models (LLMs) running via [Ollama](https://ollama.com/) and LMStudio. It allows researchers and developers to objectively compare different models on standardized programming and reasoning tasks.

## 🎯 Recent Results

- **Evaluation Accuracy**: 63% correct answers detected on HumanEval
- **Evaluation Methods**: 5 complementary approaches (normalization, AST, AI)
- **Reproducibility**: Consistent results across executions

## ✨ Features

- 🚀 **Easy to use** - Intuitive command-line interface
- 🔄 **Multi-dataset support** - Compatible with HumanEval, CruxEval, and Code-X-GLUE
- 📊 **Advanced evaluation** - Tolerates formatting, indentation, and functional equivalence differences
- 📈 **Detailed analysis** - Performance analysis scripts included
- 🧩 **Extensible** - Easy to adapt for different types of benchmarks
- 👁️ **Observability** - Integrated with Arize Phoenix for tracing and evaluation

## 🛠️ Prerequisites

- Python 3.13.5 or higher
- [Ollama](https://ollama.com/) installed and running

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/laurentvv/bench-ai-ollama-lmstudio.git
   cd bench-ai-ollama-lmstudio
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🚀 Running the benchmark

1. Make sure Ollama is running.

2. Download the model you want to test:
   ```bash
   ollama pull qwen3:14b
   ```

3. Run the benchmark with a dataset of your choice:
   ```bash
   # For HumanEval
   python run_benchmark_production.py --model_name=qwen3:14b --dataset_source=sql-console-for-openai-openai-humaneval.json
   
   # For CruxEval
   python run_benchmark_production.py --model_name=qwen3:14b --dataset_source=sql-console-for-cruxeval-org-cruxeval.json
   
   # For Code-X-GLUE
   python run_benchmark_production.py --model_name=qwen3:14b --dataset_source=sql-console-for-google-code-x-glue-ct-code-to-text.json
   ```

## 📊 Observability with Arize Phoenix

Bench AI is integrated with [Arize Phoenix](https://phoenix.arize.com/), an open-source observability library for LLMs. It allows you to trace your model's predictions and evaluate their quality in real-time.

### Phoenix Setup

Arize Phoenix runs locally and doesn't require complex configuration or Docker.

1. **Installation**: Already included in `requirements.txt`.
2. **Launch**: The benchmark script automatically launches the Phoenix app. You can access the dashboard at `http://localhost:6006`.

### Using Arize Phoenix with Bench AI

When you run the benchmark, it will automatically start a local Phoenix server and record traces of every model call and evaluation step.

```bash
python run_benchmark_production.py \
  --model_name=qwen3:14b \
  --dataset_source=sql-console-for-openai-openai-humaneval.json \
  --project="my-benchmark-project"
```

- `--project`: The project name in the Phoenix dashboard.

The Phoenix UI allows you to:
- **Trace** exactly what was sent to and received from the LLM.
- **Inspect** the spans and latency of your application.
- **Debug** RAG or agentic workflows if extended.

### Available options

| Option | Description | Default value |
|--------|-------------|---------------|
| `--model_name` | Ollama model name to test | qwen3:14b |
| `--dataset_source` | Path to dataset source file | ./sql-console-for-openai-openai-humaneval.json |
| `--dataset_type` | Dataset type (humaneval, cruxeval, code_x_glue) | auto-detected |
| `--num_responses` | Number of responses for majority voting | 1 |
| `--project` | Phoenix project name | simple_bench |
| `--temp` | Temperature for the model | 0.1 |
| `--max_tokens` | Maximum number of tokens to generate | 2048 |
| `--top_p` | Top_p value for the model | 0.95 |
| `--max_retries` | Maximum number of retries on error | 3 |
| `--custom_system_prompt` | Custom system prompt | default prompt based on dataset |

## 📊 Results Analysis

The `run_benchmark_production.py` script automatically displays detailed evaluation results, including:

- Total number of questions evaluated
- Number and percentage of correct answers
- Total execution time
- Breakdown of correct answers by evaluation method:
  - **Basic normalization**: Exact match after cleanup
  - **Advanced normalization**: Intelligent indentation handling
  - **Extreme normalization**: Removal of spaces and line breaks
  - **AST comparison**: Syntactic structure analysis
  - **AI equivalence**: Functional equivalence detection

These detailed statistics help you understand precisely the model's performance and the types of responses it generates.

## 🧩 Supported Datasets

Bench AI natively supports several popular datasets for language model evaluation:

### 📁 Datasets included in this repository

This repository contains **3 dataset files** with **30 questions each**:

- **`sql-console-for-openai-openai-humaneval.json`** - HumanEval (30 questions)
- **`sql-console-for-cruxeval-org-cruxeval.json`** - CruxEval (30 questions)  
- **`sql-console-for-google-code-x-glue-ct-code-to-text.json`** - Code-X-GLUE (30 questions)

> 💡 **For more questions**, check out the complete datasets on Hugging Face (links below)

### HumanEval

[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) is an OpenAI benchmark for evaluating code generation capabilities. It contains Python programming problems with solutions and tests.

### CruxEval

[CruxEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) is a set of programming problems designed to evaluate the reasoning capabilities of language models.

### Code-X-GLUE

[CodeXGLUE](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text) is a Google benchmark for various code-related tasks, including generating descriptions from source code.

## 🔧 Automatic Dataset Detection

Bench AI automatically detects the dataset type based on the filename or its content. You can also explicitly specify the type with the `--dataset_type` option.

## 💯 Advanced Evaluation

Bench AI uses an advanced evaluation approach that combines multiple methods to detect correct answers:

1. **Basic normalization**: Removes tags and normalizes spaces
2. **Advanced normalization**: Intelligently handles indentation and line breaks
3. **Extreme normalization**: Removes all spaces and line breaks to detect answers that differ only in formatting
4. **AST comparison**: Analyzes code syntactic structure to detect structural equivalences
5. **AI evaluation**: Uses a language model to detect functional equivalences

This approach allows for much more precise detection of correct answers, even when they differ from the expected solution in terms of style, variable names, or algorithmic approach.

## 🚀 Upcoming Features

- LMStudio support for local models
- MBPP dataset (Mostly Basic Python Problems)
- Web interface for visualizing results
- Automatic comparison between models

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
