import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

RM1_PATH = './models/rm/safetyrm1b_1epoch_v3'
RM2_PATH = './models/rm/safetyrm8b_1epoch_v3'
DATASET_PATH = './split_data/safety_data.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rm1_tokenizer = AutoTokenizer.from_pretrained(RM1_PATH)
rm2_tokenizer = AutoTokenizer.from_pretrained(RM2_PATH)

# Set pad_token if missing
rm1_tokenizer.pad_token = rm1_tokenizer.eos_token
rm2_tokenizer.pad_token = rm2_tokenizer.eos_token

rm1 = AutoModelForSequenceClassification.from_pretrained(RM1_PATH)
rm2 = AutoModelForSequenceClassification.from_pretrained(RM2_PATH)

rm1.config.pad_token_id = rm1_tokenizer.pad_token_id
rm2.config.pad_token_id = rm2_tokenizer.pad_token_id
# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    rm1 = torch.nn.DataParallel(rm1)
    rm2 = torch.nn.DataParallel(rm2)

rm1.to(DEVICE)
rm2.to(DEVICE)

def score_batch(model, tokenizer, texts, batch_size=16):
    scores = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1)
            scores.append(batch_scores.cpu())
    return torch.cat(scores).numpy()

def evaluate_rm(model, tokenizer, dataset):
    chosen_texts = [ex["chosen"][1]['content'] for ex in dataset]
    rejected_texts = [ex["rejected"][1]['content'] for ex in dataset]
    chosen_scores = score_batch(model, tokenizer, chosen_texts)
    rejected_scores = score_batch(model, tokenizer, rejected_texts)
    return chosen_scores, rejected_scores

def compute_metrics(chosen, rejected):
    correct = (chosen > rejected).astype(int)
    accuracy = np.mean(correct)
    margin = chosen - rejected
    # auc = roc_auc_score(np.ones_like(margin), margin)
    return accuracy, None, margin

def plot_results(name, chosen, rejected, margin, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.histplot(chosen, label="chosen", color="green", kde=True, stat="density")
    sns.histplot(rejected, label="rejected", color="red", kde=True, stat="density")
    plt.title(f"{name}: Score Distribution")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{name}_scores.png"))
    plt.clf()

    sns.histplot(margin, kde=True, bins=30)
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"{name}: Margin Distribution")
    plt.savefig(os.path.join(output_dir, f"{name}_margins.png"))
    plt.clf()

# Load dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Evaluate both RMs
chosen1, rejected1 = evaluate_rm(rm1, rm1_tokenizer, dataset)
chosen2, rejected2 = evaluate_rm(rm2, rm2_tokenizer, dataset)

acc1, _, margin1 = compute_metrics(chosen1, rejected1)
acc2, _, margin2 = compute_metrics(chosen2, rejected2)

OUTPUT_DIR = "./rm_eval"
plot_results("RM1", chosen1, rejected1, margin1, OUTPUT_DIR)
plot_results("RM2", chosen2, rejected2, margin2, OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump({
        "RM1": {"accuracy": acc1},
        "RM2": {"accuracy": acc2}
    }, f, indent=2)

print("✅ Evaluation complete. See rm_eval/ for plots and metrics.")