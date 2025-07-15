import os
import json
from datasets import load_dataset

# === CONFIG ===
split = "hh_rlhf_60k"  # Change to "test" or "validation" if needed
output_dir = f"./datasets/tulu-2.5-pref-data/{split}"
os.makedirs(output_dir, exist_ok=True)

# List of harmful keywords to flag "compliance" with harmful prompts
harmful_keywords = [
    "steal", "kill", "stab", "shoot", "bomb", "hack", "cheat", "exploit", "poison", 
    "kidnap", "break into", "illegal", "how to make a bomb", "how to harm", "attack", 
    "murder", "rob", "rape", "assault", "torture", "violence", "commit fraud", "scam",
    "sneak into", "without paying"
]

# Extract assistant messages from a list of dialog turns
def extract_assistant_text(messages):
    if isinstance(messages, list):
        return " ".join([m["content"] for m in messages if m.get("role") == "assistant"])
    return ""

# Check for harmful compliance in assistant response
def is_compliant_with_harmful_request(response_messages):
    assistant_text = extract_assistant_text(response_messages).lower()
    return any(kw in assistant_text for kw in harmful_keywords)

# Load dataset
dataset = load_dataset("allenai/tulu-2.5-preference-data", split=split)

# Prepare output
safety_data = []
helpful_data = []

# Classify each example
for example in dataset:
    chosen = example.get("chosen", [])
    
    if is_compliant_with_harmful_request(chosen):
        safety_data.append(example)
    else:
        helpful_data.append(example)

# Save to disk
with open(os.path.join(output_dir, "safety.json"), "w") as f:
    json.dump(safety_data, f, indent=2)

with open(os.path.join(output_dir, "helpful.json"), "w") as f:
    json.dump(helpful_data, f, indent=2)

print(f"✅ Saved {len(safety_data)} safety examples to {output_dir}/safety.json")
print(f"✅ Saved {len(helpful_data)} helpful examples to {output_dir}/helpful.json")