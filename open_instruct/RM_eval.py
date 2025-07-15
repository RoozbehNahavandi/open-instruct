# run_rewardbench.py
from reward_bench import evaluate_model
from reward_bench.tasks import get_task
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load reward model
MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# 2. Define scoring function
def score_fn(samples, prompts):
    texts = [prompt + sample for prompt, sample in zip(prompts, samples)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).tolist()
    return scores

# 3. Load safety task
task = get_task("safety")

# 4. Evaluate
results = evaluate_model(score_fn=score_fn, task=task)
print("Results:", results)