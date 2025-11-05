# ProcVQA: Process Visualization Question Answering Benchmark

**ProcVQA** is a benchmark designed to evaluate Vision-Language Models (VLMs) on their ability to understand **process visualizations**—node-link diagrams depicting frequent patterns mined from event sequence data. This benchmark enables systematic analysis of how structural properties (structure type and density) affect VLM performance.

## Features

- **118 real-world process visualizations** from 8 diverse domains (healthcare, sports, mobility, software engineering)
- **3 visualization structure types**: Trees, Graphs, and Linear Sequence Clusters
- **Controlled density levels**: Low, Medium, and High structural complexity
- **2 evaluation tasks**:
  - **Visual Data Extraction (VDE)**: Extract all node-edge relationships
  - **Visual Question Answering (VQA)**: Answer 144 expert-validated questions
- **4 reasoning categories**: Value Extraction, Sequential Reasoning, Value Aggregation, Unanswerable Detection
- **Comprehensive analysis** of 21 proprietary and open-source VLMs

### Domain Coverage
- Healthcare (Trauma, Emergency)
- Sports (Basketball, Baseball)  
- Mobility (Foursquare check-ins, VAST Challenge)
- Software Engineering (Workflow, Bug-fix activities)

## Dataset Statistics

| Visualization Type | Count | Avg Nodes | Avg Edges | Avg Unique Nodes |
|-------------------|-------|-----------|-----------|------------------|
| Trees | 34 | 11.88 | 10.88 | 5.74 |
| Graphs | 46 | 13.17 | 14.63 | 6.48 |
| Linear Sequences | 38 | 40.53 | 5.76 | 10.55 |
| **Overall** | **118** | **21.61** | **10.69** | **7.58** |

## 📁 Repository Structure

```
ProcVQA/
├── vde_images/                    # Images for Visual Data Extraction task
│   ├── trees/                     # Tree visualizations (34 images)
│   ├── graphs/                    # Graph visualizations (46 images)
│   └── linear_sequences/          # Linear sequence clusters (38 images)
│
├── vqa_images/                    # Images for Visual Question Answering task
│   ├── trees/
│   ├── graphs/
│   └── linear_sequences/
│
├── vde_ground_truth/              # Ground truth for VDE task
│   ├── trees/                     # JSON files with node-edge tuples
│   ├── graphs/
│   ├── linear_sequences/
│   └── complexity_stats.csv       # Structural density classifications
│
├── vqa_ground_truth/              # Ground truth for VQA task
│   └── VQA_mcq_with_prompts.csv   # 144 questions with answers
│
├── scripts/                       # Evaluation scripts
│   ├── vde_outputs/               # Example model outputs for VDE
│   ├── vqa_outputs/               # Example model outputs for VQA
│   └── models/                    # Model configuration files
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ProcVQA in your research, please cite:

```bibtex
@inproceedings{zinat-etal-2025-procvqa,
    title = "{P}roc{VQA}: Benchmarking the Effects of Structural Properties in Mined Process Visualizations on Vision{--}Language Model Performance",
    author = "Zinat, Kazi Tasnim  and
      Abrar, Saad Mohammad  and
      Saha, Shoumik  and
      Duppala, Sharmila  and
      Sakhamuri, Saimadhav Naga  and
      Liu, Zhicheng",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1266/",
    pages = "23316--23348",
    ISBN = "979-8-89176-335-7".
}
```

## Acknowledgements

This project builds upon previous research in visualization understanding such as Sequence Summary and visual language models.
