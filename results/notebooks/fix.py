import json
import os
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModel

# Load models for semantic similarity calculation
SEM_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
SEM_MODEL = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

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

def claim_accuracy_score(claim: str, generated_explanation: str) -> float:
    """Calculate semantic similarity between the claim and the generated explanation."""
    return semantic_similarity(claim, generated_explanation)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def recalculate_claim_accuracy(file_path: str, ground_truth_file: str) -> None:
    """Recalculate claim_accuracy_score for a given file."""
    data = load_json_file(file_path)
    ground_truth = load_json_file(ground_truth_file)

    if isinstance(data, list):
        # Handle R4C data structure
        for item, gt_item in zip(data, ground_truth):
            item['claim_accuracy_score'] = claim_accuracy_score(gt_item['claim'], item['generated_explanation'])
    elif isinstance(data, dict) and 'claim_accuracy_score' in data:
        # Handle CIVIC data structure
        new_scores = []
        for score, gt_item in zip(data['claim_accuracy_score'], ground_truth):
            new_score = claim_accuracy_score(gt_item['claim'], gt_item['generated_explanation'])
            new_scores.append(new_score)
        data['claim_accuracy_score'] = new_scores
    else:
        print(f"Unexpected data structure in {file_path}")
        return

    save_json_file(data, file_path)
    print(f"Updated {file_path}")

def process_directory(directory: str, ground_truth_dir: str) -> None:
    """Process all JSON files in a directory and its subdirectories."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                ground_truth_file = os.path.join(ground_truth_dir, relative_path)
                
                if os.path.exists(ground_truth_file):
                    recalculate_claim_accuracy(file_path, ground_truth_file)
                else:
                    print(f"Ground truth file not found for {file_path}")

# Main execution
if __name__ == "__main__":
    output_dir = "../OpenAI"
    ground_truth_dir = "../../data"

    process_directory(output_dir, ground_truth_dir)

    output_dir = "../Anthropic"
    process_directory(output_dir, ground_truth_dir)

    print("Claim accuracy score recalculation completed.")