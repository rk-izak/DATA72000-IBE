# DATA72000-IBE

## Overview

This `README` provides an overview of the `results` folder within the repository. This folder is divided into three subfolders: `Anthropic`, `OpenAI`, and `notebooks`. Each subfolder contains numerical results and files related to different models, agent types, and testing scenarios. This document will guide you through the folder structure, briefly explain the purpose of each file, and provide examples to clarify their contents.


## Folder Structure

```bash
results/
│
├── Anthropic/
│   └── claude3_haiku/
│       ├── base/
│       │   ├── assignment_test.json
│       │   ├── baseline.json
│       │   ├── full_data_noid.json
│       │   ├── missing_evidence.json
│       │   ├── mixed.json
│       │   ├── selection_test.json
│       │   └── wrong_evidence.json
│       ├── base_exp/
│       │   └── (same file structure as in base/)
│       ├── base_exp_coh/
│       │   └── (same file structure as in base/)
│       ├── coh/
│       │   └── (same file structure as in base/)
│       └── exp/
│           └── (same file structure as in base/)
│
├── OpenAI/
│   ├── gpt-4o/
│   │   └── (same file structure as in claude3_haiku/{base/exp/...}/)
│   ├── gpt-4o-mini/
│   │   └── (same file structure as in claude3_haiku/{base/exp/...}/)
│   └── gpt3.5-turbo/
│       └── (same file structure as in claude3_haiku/{base/exp/...}/)
│
└── notebooks/
    ├── anthropic_results.ipynb
    └── openai_results.ipynb
```

## General Usage
Each subfolder (`Anthropic`, `OpenAI`, and `notebooks`) is organized to provide results for files relevant to different models and scenarios. The `Anthropic` and `OpenAI` subfolders contain directories for various models and agent types, each with a set of JSON files representing different scenarios. These files include calculated evaluation metrics based on `outputs` from the agent models and test data from the `data/` folder. The `notebooks` subfolder includes Jupyter notebooks used for running evaluations. These notebooks provide all necessary steps for evaluating the models across various agent types and scenarios. Please refer to these notebooks in parallel with this `README` when accessing this folder for the first time, as they include steps necessary for replicating the results. A brief overview and guide to navigating these subfolders and their contents has been provided below.

## Subfolder Details

### Anthropic Subfolder

The `Anthropic` subfolder contains results from the Claude3 Haiku model. The results are organized by different agent types, each representing a combination of reasoning modules used for testing, almost exaclty the same as in `../outputs/` folder.

#### Structure and Files

- **claude3_haiku/**: This main folder is further divided into five subfolders, each corresponding to a specific agent type:
  - **base/**: Results for the agent with baseline reasoning modules.
  - **exp/**: Results for the agent with Explanatory Power modules.
  - **coh/**: Results for the agent with Coherence modules.
  - **base_exp/**: Results for the agent with Baseline + EXP modules.
  - **base_exp_coh/**: Results for the agent with Baseline + EXP + COH modules.

#### File Descriptions

Each agent type folder contains the following JSON files, representing different testing scenarios:

- **baseline.json**: Results for scenarios with all evidence present.
- **missing_evidence.json**: Results for scenarios where some evidence is missing.
- **wrong_evidence.json**: Results for scenarios with incorrect evidence.
- **mixed.json**: Results for scenarios with a mix of correct and missing evidence.
- **selection_test.json**: Results for the selection test scenario.
- **assignment_test.json**: Results for the assignment test scenario.
- **full_data_noid.json**: Results similar to `baseline.json` but based on a R4C data.

These files contain calculated evaluation metrics based on outputs from the agent and test data ("golden" explanations) found in the `data/` folder.

### OpenAI Subfolder

The `OpenAI` subfolder contains results generated from given OpenAI models, namely GPT-4o, GPT-4o-mini, and GPT-3.5 Turbo. The folder structure mirrors that of the `Anthropic` subfolder, with results organized by agent type.

#### Structure and Files

- **gpt-4o/**: Contains results for the GPT-4o model.
- **gpt-4o-mini/**: Contains results for a smaller version of the GPT-4o model.
- **gpt3.5-turbo/**: Contains results for the GPT-3.5 Turbo model.

Each of these subfolders includes the same set of JSON files as those found in the `Anthropic` subfolder (`assignment_test.json`, `baseline.json`, etc.), representing different testing scenarios and agent types.

#### Notebooks

- **anthropic_results.ipynb**: Notebook for testing and analyzing results from the Anthropic model (`Claude3 Haiku`).
- **openai_results.ipynb**: Notebook for testing and analyzing results from OpenAI models (`GPT-4o`, `GPT-4o-mini`, `GPT-3.5 Turbo`).

These notebooks utilize the `evaluate_explanations`, `evaluate_assignment_test`, and `evaluate_selection_test` functions, which aggregate and automate all evaluation processes. Notebooks provide a step-by-step process for testing and evaluating models based on different agent types and scenarios.


> **Note:** To understand exactly how the evaluation function works and see all the metrics calculated, refer to the `../utils/` folder. This folder contains the implementation of the evaluations functions alongside all metrics used for the scoring process.

## Example Files

---
### baseline.json, missing_evidence.json, wrong_evidence.json, mixed.json, selection_test.json, full_data_noid.json
These files all share the same structure:
```json

{
  "bertscore": [
    {
      "Precision": 0.5957040190696716,
      "Recall": 0.6904841661453247,
      "F1": 0.6396019458770752
    },
    {
      "Precision": 0.5514285564422607,
      "Recall": 0.5718935132026672,
      "F1": 0.561474621295929
    },
    ...
  ],
  "fluency": [
    0.8543736934661865,
    ...
  ],
  "explanation_accuracy": [
    0.8732708096504211,
    ...
  ],
  "claim_accuracy_score": [
    0.8816767334938049,
    ...
  ],
  "claim_support": [
    2.371371239423752,
    ...
  ],
  "fact_verification": [
    0.7871148784955343,
    ...
  ],
  "explanation_completeness": [
    0.84051114320755,
    ...
  ]
}
```
---
### assignment_test.json 
This file is almost of the exact same structure as the above, but is split into `claim_A` and `claim_B` subkeys, i.e.:

```json
{
  "claim_A": {
    "bertscore": [
      {
        "Precision": 0.5510863661766052,
        "Recall": 0.6044328212738037,
        "F1": 0.5765281319618225
      },
    ],
    "fluency": [
      0.7775493621826172,
    ],
    ...
  },
  "claim_B": {
    "bertscore": [
      {
        "Precision": 0.6738267540931702,
        "Recall": 0.6132894158363342,
        "F1": 0.642134428024292
      },
      ...
    ],
    "fluency": [
      0.8680146217346192,
      ...
    ]
    ...
  }
}

```
---
> ***Note:*** These metrics have been calculated for the 30 Test Cases chosen from each scenario. These were not chosen at random, but simply the first 30 cases were picked for each case. As such, the matching was done not via IDs, but through indexing based on positions between `outputs/` files and `data/` files. For example the first `"explanation"` in `data/CIVIC/clean/context/baseline.json` is the reference explanation for the first `"generated_explanation"` in `outputs/OpenAI/gpt-4o/base/baseline.json`. This holds true for all other tested scenarions/agents.

## Example Metrics Showcase
To better visualise the results and how these files can be used, example table and figure have been provided. More information about plots and their generation can be seen in the `../figures/` folder.

---
<div align="center">
<p><strong>Table 1: GPT-4O - Baseline Test Averages (highest score underlined + relative gain from Base Agent)</strong></p>

<table>
<tr>
<th>Metric</th><th>Base</th><th>EXP</th><th>COH</th><th>Base+EXP</th><th>Base+EXP+COH</th>
</tr>
<tr><td>BERTScore (F1)</td><td>0.5899</td><td>0.5893</td><td><u>0.5950 (+0.87%)</u></td><td>0.5877</td><td>0.5899</td></tr>
<tr><td>BERTScore (Precision)</td><td>0.5740</td><td><u>0.5836 (+1.67%)</u></td><td>0.5836</td><td>0.5775</td><td>0.5737</td></tr>
<tr><td>BERTScore (Recall)</td><td>0.6093</td><td>0.5984</td><td>0.6099</td><td>0.6015</td><td><u>0.6101 (+0.12%)</u></td></tr>
<tr><td>Fluency</td><td>0.8077</td><td>0.7955</td><td>0.7979</td><td>0.7885</td><td><u>0.8126 (+0.61%)</u></td></tr>
<tr><td>Coherence</td><td>0.9005</td><td>0.7330</td><td>0.8655</td><td><u>0.9503 (+5.54%)</u></td><td>0.8948</td></tr>
<tr><td>Explanation Accuracy</td><td>0.8356</td><td>0.8223</td><td>0.8387</td><td><u>0.8391 (+0.42%)</u></td><td>0.8326</td></tr>
<tr><td>Explanation Completeness</td><td>0.6669</td><td>0.6488</td><td>0.6656</td><td>0.6653</td><td><u>0.6724 (+0.82%)</u></td></tr>
<tr><td>Claim Accuracy</td><td>0.8515</td><td>0.8506</td><td>0.8464</td><td>0.8515</td><td><u>0.8580 (+0.76%)</u></td></tr>
<tr><td>Claim Support</td><td>1.7955</td><td>1.8075</td><td>1.9917</td><td><u>2.0633 (+14.91%)</u></td><td>1.7745</td></tr>
<tr><td>Fact Verification</td><td><u>1.0707</u></td><td>1.0268</td><td>1.0518</td><td>1.0133</td><td>1.0140</td></tr>
</table>
</div>

---

<div align="center">
    <img src="../figures/graphs/density_plots/GPT-4O/baseline_density_plots.png" alt="Density Plot" style="width: 70%; max-width: 800px;">
    <p><strong>Figure 1:</strong> Density Plots for Evaluation Metrics for GPT-4O for all 5 tested Agents (Baseline Scenario, raw scores)</p>
</div>

---
> ***Note:*** Again, for more information about specific metrics, models, etc. refer to other folders.