A framework for evaluating Vision-Language Models (VLMs) on event sequence visualizations.

!Teaser

## Overview

VLM4Vis is a comprehensive framework for evaluating how well Visual Language Models (VLMs) understand and extract information from process visualizations. The project focuses on three process visualization types:

- Tree diagrams showing event flow patterns
- Cluster of linear sequences based visualizations
- Graph  visualizations

## Project Structure

```
.
├── vde_images/
|-- vqa_images/                       # Source image files
├── GT_files/                   # Ground truth files consolidated
├── GT_JSON_files/                    # Ground truth in JSON format
├── scripts/                    # Main evaluation and processing scripts
    ├── mcq_outputs/            # VQA outputs
│   ├── outputs/                # Extraction outputs
│   ├── evaluate_scores_formatted_new.ipynb  # Main evaluation notebook
│   └── formatting_mcq_for_llm.ipynb         # Prompt formatting for models
├── preprocessing_and_eval/     # Preprocessing utilities
└── outputs/                    # Output analysis results
```

## Features

- **Multi-model Evaluation**: Support for evaluating multiple VLMs including:
  - GPT-4.1
  - Claude 3.5 (Haiku, Sonnet)
  - Llama 3.2 and Llama 4 (Maverick, Scout)
  - Gemini models
  - Qwen models
  - Gemma models

- **Comprehensive Metrics**:
  - Precision, Recall, F1 scores
  - Hallucination detection
  - Density-based analysis

- **Result Visualization**: Comparative analysis across models

## Usage

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Running Model Evaluations

1. **Configure API Keys** (for commercial models):

   Add your API keys to the appropriate configuration files or environment variables.

2. **Run the Evaluation Pipeline**:

   ```bash
   # Option 1: Use the evaluation notebook
   jupyter notebook scripts/evaluate_scores_formatted_new.ipynb
   
   # Option 2: Run from command line
   python scripts/evaluate_models.py
   ```

3. **Process Results**:

   The evaluation results will be stored in `outputs/extraction_results/` directory with:
   - Individual model results
   - Cross-model comparisons
   - Detailed hallucination analysis


## Results Analysis

The evaluation generates several output formats:

- CSV files with detailed metrics for each model
- JSON files with overall performance statistics
- Complexity analysis based on visualization characteristics
- Hallucination detection reports

Access these in the `outputs/extraction_results/` directory after running the evaluation.

## Extending the Framework

To add support for new models:

1. Create a model adapter in the appropriate script format
2. Add the model to the list of evaluated models
3. Run the evaluation pipeline with your new model

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code for your research, please cite:

```
@misc{ProcVQA,
  author = {Zinat, Kazi Tasnim},
  title = {ProcVQA: Benchmarking the Effects of Structural Properties in\\Mined Process Visualizations on Vision–Language Model Performance},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kzintas/ProcVQA}
}
```

## Acknowledgements

This project builds upon previous research in visualization understanding such as Sequence Summary and visual language models.
