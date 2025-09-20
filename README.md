Collecting workspace information# VLM4Vis

A framework for evaluating Vision-Language Models (VLMs) on event sequence visualizations.

!Teaser

## Overview

VLM4Vis is a comprehensive framework for evaluating how well Visual Language Models (VLMs) understand and extract information from process visualizations. The project focuses on three main visualization types:

- CoreFlow: Tree diagrams showing event flow patterns
- Sequence Synopsis: Cluster of linear sequences based visualizations
- SentenTree: Graph-based  visualizations

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
  - Complexity-based analysis

- **Result Visualization**: Tools to generate comparative analysis across models

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

### Working with Custom Visualizations

To evaluate models on your own visualizations:

1. Add your images to the appropriate directory in `All Visual Summaries/`
2. Create ground truth files in the proper format in GT_files
3. Update the data processing scripts as needed
4. Run the evaluation pipeline

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
@misc{VLM4Vis,
  author = {Tasnim, Zinat},
  title = {VLM4Vis: Evaluating event sequence images on open source LLMs},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/VLM4Vis}
}
```

## Acknowledgements

This project builds upon research in visualization understanding and visual language models.
