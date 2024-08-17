from typing import List, Dict

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
from evaluate import load

# Load Models
SEM_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
SEM_MODEL = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
NLI_PIPELINE = pipeline("text-classification", model="tasksource/deberta-small-long-nli")
FLUENCY_MODEL = AutoModelForCausalLM.from_pretrained("gpt2-medium")
FLUENCY_TOKENIZER = AutoTokenizer.from_pretrained("gpt2-medium")
BERTSCORE = load("bertscore")


def calculate_bertscore(candidate: str, reference: str) -> Dict[str, float]:
    """
    Calculate BERTScore for a candidate text against a reference.

    Args:
        candidate (str): The candidate text to evaluate.
        reference (str): The reference text to compare against.

    Returns:
        Dict[str, float]: A dictionary containing Precision, Recall, and F1 scores.
    """
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
    """
    Assess fluency based on perplexity.

    Args:
        generated_explanation (str): The text to assess for fluency.

    Returns:
        float: A fluency score between 0 and 1.
    """
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
    """
    Calculate semantic similarity between two texts.

    Args:
        text1 (str): The first text for comparison.
        text2 (str): The second text for comparison.

    Returns:
        float: A similarity score between -1 and 1.
    """
    inputs1 = SEM_TOKENIZER(text1, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs2 = SEM_TOKENIZER(text2, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs1 = SEM_MODEL(**inputs1)
        outputs2 = SEM_MODEL(**inputs2)
    
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).item()


def nli_score(premise: str, hypothesis: str) -> float:
    """
    Compute NLI score between premise and hypothesis.

    Args:
        premise (str): The premise text.
        hypothesis (str): The hypothesis text.

    Returns:
        float: An NLI score.
    """
    result = NLI_PIPELINE(dict(text=premise, text_pair=hypothesis))
    
    if result['label'] == 'entailment':
        return 2.5 * result['score']
    elif result['label'] == 'neutral':
        return result['score']
    else:
        return -5 * result['score']


def explanation_accuracy_score(generated_explanation: str, golden_explanation: str) -> float:
    """
    Calculate semantic similarity between generated and golden explanations.

    Args:
        generated_explanation (str): The generated explanation.
        golden_explanation (str): The golden (reference) explanation.

    Returns:
        float: A similarity score.
    """
    return semantic_similarity(generated_explanation, golden_explanation)


def claim_support_score(claim: str, generated_explanation: str) -> float:
    """
    Calculate semantic similarity between the claim and the generated explanation.

    Args:
        claim (str): The claim text.
        generated_explanation (str): The generated explanation.

    Returns:
        float: A similarity score.
    """
    return semantic_similarity(claim, generated_explanation)


def evidence_selection_score(predicted_evidence_ids: List[str], golden_evidence_ids: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 for evidence selection.

    Args:
        predicted_evidence_ids (List[str]): List of predicted evidence IDs.
        golden_evidence_ids (List[str]): List of golden (correct) evidence IDs.

    Returns:
        Dict[str, float]: A dictionary containing Precision, Recall, and F1 scores.
    """
    y_true = [1 if id in golden_evidence_ids else 0 for id in predicted_evidence_ids]
    y_pred = [1] * len(predicted_evidence_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {"Precision": precision, "Recall": recall, "F1": f1}


def coherence_score(generated_explanation: str) -> float:
    """
    Check logical coherence using NLI.

    Args:
        generated_explanation (str): The generated explanation to check for coherence.

    Returns:
        float: A coherence score.
    """
    sentences = sent_tokenize(generated_explanation)
    coherence_scores = [nli_score(sentences[i-1], sentences[i]) for i in range(1, len(sentences))]
    return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1


def fact_verification_score(generated_explanation: str, evidence_texts: List[str]) -> float:
    """
    Verify the factual accuracy of the generated explanation against the provided evidence.

    Args:
        generated_explanation (str): The generated explanation to verify.
        evidence_texts (List[str]): List of evidence texts to compare against.

    Returns:
        float: A fact verification score.
    """
    verification_scores = []
    for evidence in evidence_texts:
        evidence_sentences = sent_tokenize(evidence)
        evidence_score = max(nli_score(sent, generated_explanation) for sent in evidence_sentences)
        verification_scores.append(evidence_score)
    return sum(verification_scores) / len(verification_scores) if verification_scores else 0


def explanation_completeness_score(generated_explanation: str, golden_explanation: str) -> float:
    """
    Evaluate how much of the golden explanation's content is covered in the generated one.

    Args:
        generated_explanation (str): The generated explanation to evaluate.
        golden_explanation (str): The golden (reference) explanation.

    Returns:
        float: A completeness score.
    """
    golden_sentences = sent_tokenize(golden_explanation)
    generated_sentences = sent_tokenize(generated_explanation)
    
    coverage_scores = []
    for golden_sent in golden_sentences:
        sent_scores = [semantic_similarity(golden_sent, gen_sent) for gen_sent in generated_sentences]
        coverage_scores.append(max(sent_scores))
    
    return sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0


def evidence_relevance_score(claim: str, selected_evidence: List[str]) -> float:
    """
    Evaluate the relevance of selected evidence to the claim.

    Args:
        claim (str): The claim text.
        selected_evidence (List[str]): List of selected evidence texts.

    Returns:
        float: An evidence relevance score.
    """
    relevance_scores = []
    for evidence in selected_evidence:
        evidence_sentences = sent_tokenize(evidence)
        evidence_score = max(semantic_similarity(claim, sent) for sent in evidence_sentences)
        relevance_scores.append(evidence_score)
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0


def claim_evidence_alignment_score(claim: str, selected_evidence: List[str]) -> float:
    """
    Evaluate how well the selected evidence aligns with the claim.

    Args:
        claim (str): The claim text.
        selected_evidence (List[str]): List of selected evidence texts.

    Returns:
        float: A claim-evidence alignment score.
    """
    alignment_scores = []
    for evidence in selected_evidence:
        evidence_sentences = sent_tokenize(evidence)
        evidence_score = max(nli_score(sent, claim) for sent in evidence_sentences)
        alignment_scores.append(evidence_score)
    return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
