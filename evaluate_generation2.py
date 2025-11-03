# evaluate_generation.py
import os
import json
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from generate_email import generate_personalized_email


# CONFIGURATION
EVAL_FILE = os.path.join("data", "valid.jsonl")
RESULT_DIR = "evaluation_results"
os.makedirs(RESULT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(RESULT_DIR, f"eval_results_{timestamp}.jsonl")

# MODEL SETUP
embedder = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_metrics(generated, reference):
    """Compute multiple text similarity metrics."""
    # Semantic similarity
    emb1 = embedder.encode(generated, convert_to_tensor=True)
    emb2 = embedder.encode(reference, convert_to_tensor=True)
    cosine_sim = util.cos_sim(emb1, emb2).item()

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)

    # ROUGE-L
    rougeL = scorer.score(reference, generated)["rougeL"].fmeasure

    # BERTScore
    P, R, F1 = bertscore([generated], [reference], lang="en", verbose=False)
    bert_f1 = float(F1.mean())

    return {
        "semantic_similarity": round(cosine_sim, 4),
        "bleu": round(bleu, 4),
        "rougeL": round(rougeL, 4),
        "bertscore_F1": round(bert_f1, 4),
    }


def evaluate_model(dataset_path=EVAL_FILE):
    """Run model evaluation on dataset and compute average metrics."""
    if not os.path.exists(dataset_path):
        print(" Dataset not found. Expected:", dataset_path)
        return

    print(f" Evaluating model on {dataset_path} ...\n")
    dataset = load_dataset("json", data_files={"valid": dataset_path})["valid"]

    all_metrics = {k: [] for k in ["semantic_similarity", "bleu", "rougeL", "bertscore_F1"]}

    for idx, example in enumerate(dataset):
        fields = example["input"]
        reference = example["output"]
        generated = generate_personalized_email(fields)

        metrics = compute_metrics(generated, reference)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

        result = {
            "index": idx,
            "input": fields,
            "generated": generated,
            "reference": reference,
            **metrics
        }

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

        print(f"[{idx+1}/{len(dataset)}] Similarity={metrics['semantic_similarity']}  BLEU={metrics['bleu']}  ROUGE-L={metrics['rougeL']}")

    # Averages
    avg_metrics = {k: round(sum(v)/len(v), 4) for k, v in all_metrics.items()}
    summary_path = os.path.join(RESULT_DIR, f"summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "averages": avg_metrics, "total_samples": len(dataset)}, f, indent=4)

    print("\n Evaluation completed.")
    print(f"Average metrics: {avg_metrics}")
    print(f"Detailed results saved to: {OUTPUT_FILE}")
    print(f"Summary file: {summary_path}")
    return avg_metrics
