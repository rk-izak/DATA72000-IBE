# DATA72000-IBE

## Overview

This README provides an overview of the `figures` folder within the repository. This folder is structured to organize various graphical representations and architectural diagrams produced for the project and/or report. This document will guide you through the folder structure, describe the contents of subfolders, and explain the main purposes of the files.

```bash
figures/
│
├── architectures/
│   ├── templates/
│   │   ├── mas.drawio
│   │   ├── r4c.drawio
│   │   ├── rag.drawio
│   │   ├── rag_task.drawio
│   │   ├── sas.drawio
│   │   └── sda.drawio
│   │
│   └── (same filenames as in templates, with `.drawio.png` extension)
│
└── graphs/
    ├── boxplots/
    │   ├── CLAUDE3_HAIKU/
    │   │   ├── assignment_test_boxplots.png
    │   │   ├── baseline_boxplots.png
    │   │   ├── full_data_noid_boxplots.png
    │   │   ├── missing_evidence_boxplots.png
    │   │   ├── mixed_boxplots.png
    │   │   ├── selection_test_boxplots.png
    │   │   └── wrong_evidence_boxplots.png
    │   ├── GPT-4O/
    │   ├── GPT-4O-MINI/
    │   └── GPT3.5-TURBO/
    │       └── (same file structure as CLAUDE3_HAIKU)
    │
    ├── correlation_heatmaps/
    ├── density_plots/
    ├── heatmaps/
    └── radar_plots/
        └── (same folder structure as boxplots for each model)
    │
    └── performance_comparison/
        ├── assignment_test_performance_comparison.png
        ├── baseline_performance_comparison.png
        ├── full_data_noid_performance_comparison.png
        ├── missing_evidence_performance_comparison.png
        ├── mixed_performance_comparison.png
        ├── selection_test_performance_comparison.png
        └── wrong_evidence_performance_comparison.png

```

## Subfolder Details

### Architectures Subfolder

The `architectures` subfolder contains diagrams representing different architectures or workflows used within the project. It includes both the original `.drawio` files and their corresponding exported `.png` images.

#### Structure and Files

- **templates/**: This folder contains the original `.drawio` files that define the architectural diagrams:
  - `mas.drawio`, `r4c.drawio`, `rag.drawio`, `rag_task.drawio`, `sas.drawio`, `sda.drawio`
  - Editable in [draw.io](draw.io) and represent used and future workflows/agents.
  
- **Exported Images**: The `.png` files are exported versions of the `.drawio` diagrams for easy viewing:
  - Corresponding images of the `.drawio` files: `mas.drawio.png`, `r4c.drawio.png`, `rag.drawio.png`, `rag_task.drawio.png`, `sas.drawio.png`, `sda.drawio.png`
  - Used for documentation and reporting.

### Graphs Subfolder

The `graphs` subfolder contains visual representations of various performance metrics and comparisons across different models and scenarios. This subfolder is further divided into specific types of visualizations.

#### Structure and Files

- **boxplots/**, **correlation_heatmaps/**, **density_plots/**, **heatmaps/**, **radar_plots/**: 
  - Each of these folders contains subfolders corresponding to different models (`CLAUDE3_HAIKU`, `GPT-4O`, `GPT-4O-MINI`, `GPT3.5-TURBO`).
  - Within each model-specific folder are `.png` files representing performance metrics under different testing scenarios: `assignment_test_*.png`, `baseline_*.png`, `full_data_noid_*.png`, `missing_evidence_*.png`, `mixed_*.png`, `selection_test_*.png`, `wrong_evidence_*.png`
  - The suffix indicates the type of visualization (e.g., boxplots, radar plots).

- **performance_comparison/**: 
  - This folder includes aggregate comparison plots that evaluate model performance across all language models.
  - Files follow a similar naming convention (`*_performance_comparison.png`) and represent aggregate comparisons for different test scenarios.

### Example Files

Below are examples of the contents within some of the subfolders to illustrate them:

#### Example from `boxplots/CLAUDE3_HAIKU/`

- **baseline_boxplots.png**: Shows boxplot distributions for the baseline scenario, comparing metrics like BERTScore, fluency, and explanation accuracy across various agents.

---

<div align="center">
    <img src="graphs/boxplots/CLAUDE3_HAIKU/baseline_boxplots.png" alt="Density Plot" style="width: 70%; max-width: 800px;">
    <p><strong>Figure 1:</strong> Boxplots for Evaluation Metrics for CLAUDE3 HAIKU for all 5 tested Agents (Baseline Test)</p>
</div>

---


#### Example from `performance_comparison/`

- **assignment_test_performance_comparison.png**: A performance comparison plot aggregating all models under the assignment test scenario, showing relative performance across metrics.

---

<div align="center">
    <img src="graphs/performance_comparison/assignment_test_performance_comparison.png" alt="Density Plot" style="width: 70%; max-width: 800px;">
    <p><strong>Figure 2:</strong> Performance Comparison for Evaluation Metrics for all LLMs, for all 5 tested Agents (Assignment Test)</p>
</div>

---
> ***Note:*** The `graphs/` folder contains very roughly processed results as the graphs are meant to showcase what the data looks like and how can be used, and are meant to support the findings further rather than be discussion points on their own.

### Jupyter Notebooks

The `graphs.ipynb` and `tables.ipynb` notebooks within automate the generation of aforementioned graphs and tables:

- **graphs.ipynb**: Generates visualizations for various metrics across different models and scenarios, saving them into the appropriate folders.

- **tables.ipynb**: Processes results and generates formatted tables for easy comparison of performance metrics across different agents and scenarios.

> ***Note:*** For more detailed information on the generation and analysis of these figures, please refer to the respective notebooks.
