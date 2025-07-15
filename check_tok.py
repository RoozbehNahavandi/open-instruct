from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Configuration ===
model_path = "meta-llama/Llama-2-13b-hf"
# model_path = "models/base_ppo/Tulu-V2.5-7B_V1"

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "Can you explain why the sky is blue?"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path)

# === Load model with ignore mismatch ===
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=None,
    ignore_mismatched_sizes=True  # ✅ handles vocab size mismatch
)

model.to('cuda')
# === Resize embeddings if needed (optional) ===
# model.resize_token_embeddings(len(tokenizer))  # Only if you suspect mismatched embedding

# === Prepare input ===
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# === Generate response ===
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

# === Decode and print ===
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Generated ===")
print(response)