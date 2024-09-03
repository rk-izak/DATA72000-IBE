# DATA72000-IBE

This `README` provides an overview of the `utils` folder within the repository. This folder contains utility scripts, notebooks, and test files essential for evaluating tested models. This folder includes one subfolder, `scorers_test`, which is used for unit testing various scoring metrics for better selection of underlying models. This document will guide you through the folder structure, briefly explain the purpose of each file, and provide examples to clarify contents.

## Folder Structure

```bash
utils/
│
├── scorers_test/
│   ├── scorers_test.ipynb
│   └── scorers_unit_test.json
│
├── auto_tester.py
├── keys.json
├── playground.ipynb
└── utils.py
```

## General Usage

The `utils` folder is organized to provide utility functions, API key management, scoring metrics, testing scripts, and notebooks for interacting with `Self-Discover` and `Self-RAG` agents. The `scorers_test` subfolder is specifically focused on unit testing the scoring metrics implemented in `utils.py`. Below is a detailed overview of the files and their purposes within this folder.

## Subfolder Details

### scorers_test Subfolder

The `scorers_test` subfolder is designed to test the evaluation metrics defined in `utils.py`. It contains a Jupyter notebook and a JSON file with test cases and expected results (high, low, medium).

#### Structure and Files

- **scorers_test.ipynb**: A Jupyter notebook used for unit testing all metrics defined in `utils/utils.py`. It loads the `scorers_unit_test.json` file and runs a series of tests to ensure that the scoring functions perform as expected. Also can be used to test different `HuggingFace` models by adjusting them in `utils/utils.py`.

- **scorers_unit_test.json**: This JSON file contains simple test cases alongside expected results. It is used to validate the accuracy and reliability of the scoring metrics. Based mostly on the style and formatting of `CIViC` data,

### Main Files

#### utils.py

The `utils.py` file defines a set of utility functions and evaluation metrics for assessing the quality of generated explanations in natural language processing tasks. Here's an overview of its main components:

- **Scoring Functions**:
  - **BERTScore**
  - **Fluency (via perplexity)**
  - **Semantic similarity**
  - **Natural Language Inference (NLI)**
  - **Explanation accuracy**
  - **Claim accuracy and support**
  - **Evidence selection**
  - **Coherence**
  - **Fact verification**
  - **Explanation completeness**
  - **Evidence relevance**
  - **Claim-evidence alignment**

- **Utility Functions**:
  - **load_json_file**: Loads data from a JSON file.
  - **save_json_file**: Saves data to a JSON file.
  - **set_api_key**: Sets API keys as environment variables based on the `keys.json` file.

- **Main Evaluation Functions**:
  - **evaluate_explanations**: Evaluates explanations using various metrics for CIViC or R4C datasets (`explanation` task).
  - **evaluate_selection_test**: Evaluates explanations for selection tests (`selection` task). Modified `evaluate_explanations` function.
  - **evaluate_assignment_test**: Evaluates explanations for assignment tests (`assignment` task). Modified `evaluate_explanations` function.


The main eval. functions aggregate and calculate multiple metrics for generated explanations and save the results to a JSON file. For more information about each, please refer to the main `report.pdf` or specific function in the `utils.py` file.

As the metrics evaluate generated text against source text, the following language models accessed via `HuggingFace` website were used in the process:

<div align="center">

| Model Name                                                                 | Description                                    |
|----------------------------------------------------------------------------|------------------------------------------------|
| [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Used for Sentence Similarity (Semantics) based Metrics |
| [tasksource/deberta-small-long-nli](https://huggingface.co/tasksource/deberta-small-long-nli) | Used for Natural Language Inference (NLI) based Metrics |
| [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)         | Used for Fluency Metric                        |
| [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | Used for BERTSCORE                             |


</div>


> ***Note:*** Not all metrics found in the `utils.py` were used for the final evaluation or report, but were kept for future reference.

#### auto_tester.py

The `auto_tester.py` file defines an `AutomaticTester` class for automatically testing Agents on various tasks. Here's a summary of its main components and functionalities:

- **Initialization**: Accepts agents for solving tasks and retrieving evidence, file paths for input and output, and configuration parameters.
- **Data Formatting**: Includes methods to format entries from the input file based on the task type (explanation, selection, or assignment).
- **Question Generation**: Generates and extracts questions related to the given task.
- **Explanation and Evidence Extraction**: Extracts explanations and evidence (if needed) from generated text for different task types.
- **Test Running**: The `run_test` method processes entries, optionally uses RAG, solves defined tasks, and extracts results.
- **RAG Processing**: Generates questions, retrieves relevant evidence, and updates the task entry with unique RAG evidence if enabled.
- **Result Handling**: Saves the extracted results to the specified output file.
- **Main Run Method**: Manages the entire testing process, handling different task types and configurations.

This class automates the process of testing different Agents (Base, EXP, COH, etc.) on various tasks, with the flexibility to include RAG and process different types of tasks (explanation, assignment, selection). It provides detailed logging throughout the process for monitoring and debugging.

#### keys.json
The `keys.json` file stores API keys for various services (OpenAI, Anthropic, Langchain, etc.). For first-time use, it's recommended to add your API keys to this file since the `set_api_key` function in `utils.py` relies on it to set environment variables. This function is used in many notebooks throughout the repository for consistency. If you choose to skip this step, please ignore any cells containing `set_api_key` and set required keys in your preferred way.

#### playground.ipynb

The `playground.ipynb` notebook is used for interactively testing `Self-Discover` and `Self-RAG` agents with simple examples provided for each. For more information on these agents, users should refer to the `models/` folder.

## Example Usage

### scorers_test/scorers_test.ipynb

To run the unit tests for the scoring metrics, open the `scorers_test.ipynb` notebook in Jupyter. The notebook currently loads test cases from `scorers_unit_test.json`, but any other file respecting the original JSON structure will work as well.

### keys.json

To set your API keys, open the `keys.json` file and replace the placeholder text with your actual API keys:

```json
    {
        "name": "OpenAI",
        "key": "YOUR_API_KEY"
    },
    {
        "name": "LangChain",
        "key": "YOUR_API_KEY"
    },
    {
        "name": "Anthropic",
        "key": "YOUR_API_KEY"
    }
```
Currently, the `set_api_key` function only accepts these three services, but can be easily expanded for different APIs (like `Tavily`) if needed.

Alternatively, you can set these keys as environment variables using your operating system's method for that.

### utils.py

You can directly use the utility functions and evaluation metrics defined in `utils.py` in your own scripts and notebooks. For example, to evaluate explanations using the provided metrics, you might do something like this:

```python
from utils import set_api_key, calculate_bertscore, nli_score

set_api_key(file_path="keys.json", api_name="OpenAI")
set_api_key(file_path="keys.json", api_name="Anthropic")

your_generated_explanation = "Dogs dislike cats"
your_reference_explanation = "Cats dislike dogs"

result = calculate_bertscore(your_generated_explanation, your_reference_explanation)
print(result)

premise = "Garfield hates mondays"
hypothesis = "Garfield prefers weekends"

result = nli_score(premise, hypothesis)
print(result)
```

For more information about usage, or example cases/tests, please refer either to specific metricsi in `utils.py` or `scorers_test/` folder and its files.

### auto_tester.py

To use the `AutomaticTester` class, you would typically do the following:

```python
from auto_tester import AutomaticTester
tester = AutomaticTester(
    sda_agent=SELF_DISCOVER_AGENT_INSTANCE,
    rag_agent=RAG_AGENT_INSTANCE,
    input_file_path=INPUT_PATH,
    output_file_path=OUTPUT_PATH,
    num_questions=2,
    extract_model='gpt-4o-mini'
)

tester.run_test(task_type=TASK_TYPE, use_rag=BOOLEAN, num_examples=NUM_EXAMPLES)
```

However, this class was almost exclusively created for the purposes of automation for this specific project and as such is not very elastic in terms of its use. The following should be respected:

- `sda_agent` must be an instance of the `SelfDiscovery` class (found in `models/`).
- `rag_agent` must be an instance of the `SelfRag` class (found in `models/`).
- `input_file_path` should point to one of the 7 testing scenarios in the `data` folder.
- `output_file_path` can be any path where outputs need to be saved.
- `num_questions` specifies how many questions the SDA agent will ask the RAG agent.
- `extract_model` must be an OpenAI LLM, used for Regex assisted answer extraction.
- `task_type` must be one of the following: `explanation`, `selection`, or `assignment`, must comply with `input_file_path`.
- `use_rag` can be either `True` or `False`.
- `num_examples` determines how many examples from the input file to use (applies list slicing like `[:num_examples]`).

For more examples, please see `../outputs/notebooks/` folder and its usage.

> ***Note***: For more detailed information about specific prompts, functions, classes and their parameters, refer to the docstrings in each script.