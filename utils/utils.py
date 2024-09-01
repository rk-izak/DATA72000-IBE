import json
from typing import List, Dict, Any
import os
import getpass

import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
from evaluate import load

# Load models
SEM_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
SEM_MODEL = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
NLI_PIPELINE = pipeline("text-classification", model="tasksource/deberta-small-long-nli")
FLUENCY_MODEL = AutoModelForCausalLM.from_pretrained("gpt2-medium")
FLUENCY_TOKENIZER = AutoTokenizer.from_pretrained("gpt2-medium")
BERTSCORE = load("bertscore")

def calculate_bertscore(candidate: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for a candidate text against a reference."""
    if not candidate:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    
    results = BERTSCORE.compute(
        predictions=[candidate],
        references=[reference],
        lang="en",
        model_type="facebook/bart-large-mnli"
    )
    return {
        "Precision": results['precision'][0],
        "Recall": results['recall'][0],
        "F1": results['f1'][0]
    }

def fluency_score(generated_explanation: str) -> float:
    """Assess fluency based on perplexity."""
    if not generated_explanation:
        return 0.0
    
    inputs = FLUENCY_TOKENIZER(
        generated_explanation,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    with torch.no_grad():
        outputs = FLUENCY_MODEL(**inputs, labels=inputs["input_ids"])
    perplexity = torch.exp(outputs.loss).item()
    return max(0, 1 - perplexity / 100)

def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    inputs1 = SEM_TOKENIZER(text1, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs2 = SEM_TOKENIZER(text2, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs1 = SEM_MODEL(**inputs1)
        outputs2 = SEM_MODEL(**inputs2)

    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).item()

def nli_score(premise: str, hypothesis: str) -> float:
    """Compute NLI score between premise and hypothesis."""
    if not premise or not hypothesis:
        return 0.0
    
    result = NLI_PIPELINE(dict(text=premise, text_pair=hypothesis))

    if result['label'] == 'entailment':
        return 2.5 * result['score']
    elif result['label'] == 'neutral':
        return result['score']
    else:
        return -5 * result['score']

def explanation_accuracy_score(generated_explanation: str, golden_explanation: str) -> float:
    """Calculate semantic similarity between generated and golden explanations."""
    return semantic_similarity(generated_explanation, golden_explanation)

def claim_accuracy_score(claim: str, generated_explanation: str) -> float:
    """Calculate semantic similarity between the claim and the generated explanation."""
    return semantic_similarity(claim, generated_explanation)

def claim_support_score(claim: str, generated_explanation: str) -> float:
    """Calculate NLI Score between the claim and the generated explanation."""
    return nli_score(generated_explanation, claim)

def evidence_selection_score(predicted_evidence_ids: List[str], golden_evidence_ids: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 for evidence selection."""
    if not predicted_evidence_ids:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    
    y_true = [1 if id in golden_evidence_ids else 0 for id in predicted_evidence_ids]
    y_pred = [1] * len(predicted_evidence_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {"Precision": precision, "Recall": recall, "F1": f1}

def coherence_score(generated_explanation: str) -> float:
    """Check logical coherence using NLI."""
    if not generated_explanation:
        return 0.0
    
    sentences = sent_tokenize(generated_explanation)
    if len(sentences) < 2:
        return 1.0  # A single sentence is considered coherent
    
    coherence_scores = [nli_score(sentences[i-1], sentences[i]) for i in range(1, len(sentences))]
    return sum(coherence_scores) / len(coherence_scores)

def fact_verification_score(generated_explanation: str, evidence_texts: List[str]) -> float:
    """Verify the factual accuracy of the generated explanation against the provided evidence."""
    if not generated_explanation or not evidence_texts:
        return 0.0
    
    verification_scores = []
    for evidence in evidence_texts:
        evidence_sentences = sent_tokenize(evidence)
        evidence_score = max(nli_score(sent, generated_explanation) for sent in evidence_sentences)
        verification_scores.append(evidence_score)
    return sum(verification_scores) / len(verification_scores)

def explanation_completeness_score(generated_explanation: str, golden_explanation: str) -> float:
    """Evaluate how much of the golden explanation's content is covered in the generated one."""
    if not generated_explanation:
        return 0.0
    
    golden_sentences = sent_tokenize(golden_explanation)
    generated_sentences = sent_tokenize(generated_explanation)

    coverage_scores = []
    for golden_sent in golden_sentences:
        sent_scores = [semantic_similarity(golden_sent, gen_sent) for gen_sent in generated_sentences]
        coverage_scores.append(max(sent_scores))

    return sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0

def evidence_relevance_score(claim: str, selected_evidence: List[str]) -> float:
    """Evaluate the relevance of selected evidence to the claim."""
    if not selected_evidence:
        return 0.0
    
    relevance_scores = []
    for evidence in selected_evidence:
        evidence_sentences = sent_tokenize(evidence)
        evidence_score = max(semantic_similarity(claim, sent) for sent in evidence_sentences)
        relevance_scores.append(evidence_score)
    return sum(relevance_scores) / len(relevance_scores)

def claim_evidence_alignment_score(claim: str, selected_evidence: List[str]) -> float:
    """Evaluate how well the selected evidence aligns with the claim."""
    if not selected_evidence:
        return 0.0
    
    alignment_scores = []
    for evidence in selected_evidence:
        evidence_sentences = sent_tokenize(evidence)
        evidence_score = max(nli_score(sent, claim) for sent in evidence_sentences)
        alignment_scores.append(evidence_score)
    return sum(alignment_scores) / len(alignment_scores)

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def set_api_key(file_path: str = "./utils/keys.json", api_name: str = "") -> None:
    """Set the API key as an environment variable based on the provided API name."""
    try:
        with open(file_path, 'r') as file:
            api_keys = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{file_path}'.")
        return

    api_dict = {api["name"].lower(): api["key"] for api in api_keys}
    api_name = api_name.lower()

    if api_name in api_dict:
        env_var_name = f"{api_name.upper()}_API_KEY"
        os.environ[env_var_name] = api_dict[api_name]

        if api_name == "langchain":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

        if not os.environ.get(env_var_name):
            os.environ[env_var_name] = getpass.getpass(f"{env_var_name}: ")
    else:
        print(f"API name '{api_name}' not found in the provided file.")


def evaluate_explanations(ground_truth_file: str, response_file: str, save_file: str, type: str='civic') -> None:
    """Evaluate explanations using various metrics and save the results."""
    ground_truth_data = load_json_file(ground_truth_file)
    response_data = load_json_file(response_file)

    metrics = {
        "bertscore": [],
        "fluency": [],
        "explanation_accuracy": [],
        "claim_accuracy_score": [],
        "claim_support": [],
        "coherence": [],
        "fact_verification": [],
        "explanation_completeness": [],
    }

    for gt, resp in zip(ground_truth_data, response_data):

        # Calculate chosen metrics
        metrics["bertscore"].append(calculate_bertscore(resp["generated_explanation"], gt["explanation"]))
        metrics["fluency"].append(fluency_score(resp["generated_explanation"]))
        metrics["explanation_accuracy"].append(explanation_accuracy_score(resp["generated_explanation"], gt["explanation"]))
        metrics["claim_accuracy_score"].append(claim_accuracy_score(gt["claim"], resp["generated_explanation"]))
        metrics["claim_support"].append(claim_support_score(gt["claim"], resp["generated_explanation"]))
        metrics["coherence"].append(coherence_score(resp["generated_explanation"]))
        if type == 'civic':
            metrics["fact_verification"].append(fact_verification_score(resp["generated_explanation"], [e["description"] for e in gt["evidence"]]))
        elif type == 'r4c':
            metrics["fact_verification"].append(fact_verification_score(resp["generated_explanation"], [e["description"] for e in gt["evidence_golden"]]))
        metrics["explanation_completeness"].append(explanation_completeness_score(resp["generated_explanation"], gt["explanation"]))

    save_json_file(metrics, save_file)
    print(f"Evaluation complete. Results saved to {save_file}")

def evaluate_assignment_test(ground_truth_file: str, response_file: str, save_file: str) -> None:
    """Evaluate explanations for assignment test using various metrics and save the results."""
    ground_truth_data = load_json_file(ground_truth_file)
    response_data = load_json_file(response_file)

    metrics = {
        "claim_A": {
            "bertscore": [],
            "fluency": [],
            "explanation_accuracy": [],
            "claim_accuracy_score": [],
            "claim_support": [],
            "coherence": [],
            "fact_verification": [],
            "explanation_completeness": [],
        },
        "claim_B": {
            "bertscore": [],
            "fluency": [],
            "explanation_accuracy": [],
            "claim_accuracy_score": [],
            "claim_support": [],
            "coherence": [],
            "fact_verification": [],
            "explanation_completeness": [],
        }
    }

    for gt, resp in zip(ground_truth_data, response_data):
        for claim in ['A', 'B']:
            claim_key = f"claim_{claim}"
            # Calculate chosen metrics
            metrics[claim_key]["bertscore"].append(calculate_bertscore(resp[claim_key]["generated_explanation"], gt[f"explanation_{claim}"]))
            metrics[claim_key]["fluency"].append(fluency_score(resp[claim_key]["generated_explanation"]))
            metrics[claim_key]["explanation_accuracy"].append(explanation_accuracy_score(resp[claim_key]["generated_explanation"], gt[f"explanation_{claim}"]))
            metrics[claim_key]["claim_accuracy_score"].append(claim_accuracy_score(gt[claim_key], resp[claim_key]["generated_explanation"]))
            metrics[claim_key]["claim_support"].append(claim_support_score(gt[claim_key], resp[claim_key]["generated_explanation"]))
            metrics[claim_key]["coherence"].append(coherence_score(resp[claim_key]["generated_explanation"]))
            metrics[claim_key]["fact_verification"].append(fact_verification_score(resp[claim_key]["generated_explanation"], [e["description"] for e in gt[f"evidence_{claim}"]]))
            metrics[claim_key]["explanation_completeness"].append(explanation_completeness_score(resp[claim_key]["generated_explanation"], gt[f"explanation_{claim}"]))

    save_json_file(metrics, save_file)
    print(f"Evaluation complete. Results saved to {save_file}")

def evaluate_selection_test(ground_truth_file: str, response_file: str, save_file: str) -> None:
    """Evaluate explanations for selection test using various metrics and save the results."""
    ground_truth_data = load_json_file(ground_truth_file)
    response_data = load_json_file(response_file)

    metrics = {
        "bertscore": [],
        "fluency": [],
        "explanation_accuracy": [],
        "claim_accuracy_score": [],
        "claim_support": [],
        "coherence": [],
        "fact_verification": [],
        "explanation_completeness": [],
    }

    for gt, resp in zip(ground_truth_data, response_data):
        # Calculate chosen metrics
        metrics["bertscore"].append(calculate_bertscore(resp["generated_explanation"], gt["explanation_A"]))
        metrics["fluency"].append(fluency_score(resp["generated_explanation"]))
        metrics["explanation_accuracy"].append(explanation_accuracy_score(resp["generated_explanation"], gt["explanation_A"]))
        metrics["claim_accuracy_score"].append(claim_accuracy_score(gt["claim_A"], resp["generated_explanation"]))
        metrics["claim_support"].append(claim_support_score(gt["claim_A"], resp["generated_explanation"]))
        metrics["coherence"].append(coherence_score(resp["generated_explanation"]))
        metrics["fact_verification"].append(fact_verification_score(resp["generated_explanation"], [e["description"] for e in gt["evidence_A"]]))
        metrics["explanation_completeness"].append(explanation_completeness_score(resp["generated_explanation"], gt["explanation_A"]))

    save_json_file(metrics, save_file)
    print(f"Evaluation complete. Results saved to {save_file}")
