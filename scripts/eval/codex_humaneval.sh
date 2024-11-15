# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

# Evaluating llama 7B model using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/llama_7B_temp_0_1 \
    --model ../hf_llama2_models/7B/ \
    --tokenizer ../hf_llama2_models/7B/ \
    --use_vllm

echo "Finished evaluation - model=llama-7B, temp=0.1, pass@1,5,10,20"


# Evaluating llama 7B model using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/llama_7B_temp_0_8 \
    --model ../hf_llama2_models/7B/ \
    --tokenizer ../hf_llama2_models/7B/ \
    --use_vllm

echo "Finished evaluation - model=llama-7B, temp=0.8, pass@10"



# Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/tulu_7B_temp_0_1 \
    --model allenai/tulu-2-7b \
    --tokenizer allenai/tulu-2-7b \
    --use_vllm

echo "Finished evaluation - model=allenai/tulu-v2-7B, temp=0.1, pass@1,5,10,20"


# Evaluating tulu 7B model using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/tulu_7B_temp_0_8 \
    --model allenai/tulu-2-7b \
    --tokenizer allenai/tulu-2-7b \
    --use_vllm

echo "Finished evaluation - model=allenai/tulu-v2-7B, temp=0.8, pass@10"


# Evaluating llama 7B finetuned on tuluv2 using SFT model using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/SFT_temp_0_1 \
    --model output/tulu_v2_7B \
    --tokenizer output/tulu_v2_7B \
    --use_vllm

echo "Finished evaluation - model=llama7B-sft, temp=0.1, pass@1,5,10,20"


# Evaluating llama 7B finetuned on tuluv2 using SFT model using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/SFT_temp_0_8 \
    --model output/tulu_v2_7B \
    --tokenizer output/tulu_v2_7B \
    --use_vllm

echo "Finished evaluation - model=llama7B-sft, temp=0.8, pass@10"


# Evaluating llama 7B finetuned on tuluv2 using DPO-SFT model using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/DPO_SFT_temp_0_1 \
    --model output/dpo_7b_recreate2 \
    --tokenizer output/dpo_7b_recreate2 \
    --use_vllm

echo "Finished evaluation - model=llama7B-dposft, temp=0.1, pass@1,5,10,20"


# Evaluating llama 7B finetuned on tuluv2 using DPO-SFT model using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/DPO_SFT_temp_0_8 \
    --model output/dpo_7b_recreate2 \
    --tokenizer output/dpo_7b_recreate2 \
    --use_vllm

echo "Finished evaluation - model=llama7B-dposft, temp=0.8, pass@10"


# Evaluating llama 7B finetuned on tuluv2 using DPO-lora model using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/DPO_lora_temp_0_1 \
    --model output/dpo_7b_recreate2_lora \
    --tokenizer output/dpo_7b_recreate2_lora \
    --use_vllm

echo "Finished evaluation - model=llama7B-dpolora, temp=0.1, pass@1,5,10,20"


# Evaluating llama 7B finetuned on tuluv2 using DPO-lora model using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/DPO_lora_temp_0_8 \
    --model output/dpo_7b_recreate2_lora \
    --tokenizer output/dpo_7b_recreate2_lora \
    --use_vllm

echo "Finished evaluation - model=llama7B-dpolora, temp=0.8, pass@10"


# # Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score with chat format via HumanEvalPack
# # And use the HumanEval+ data for more rigorous evaluation.
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEvalPlus-OriginFmt.jsonl.gz  \
#     --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/tulu_7B_temp_0_1 \
#     --model ../checkpoints/tulu_7B/ \
#     --tokenizer ../checkpoints/tulu_7B/ \
#     --use_vllm

# # you can also use humaneval+ without the HumanEvalPack data
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEvalPlus-OriginFmt.jsonl.gz  \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/tulu_7B_temp_0_1 \
#     --model ../checkpoints/tulu_7B/ \
#     --tokenizer ../checkpoints/tulu_7B/ \
#     --use_vllm

# # Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score without chat format
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/tulu_7B_temp_0_1_nochat \
#     --model ../checkpoints/tulu_7B/ \
#     --tokenizer ../checkpoints/tulu_7B/ \
#     --use_vllm


# # Evaluating tulu 7B model using temperature 0.8 to get the pass@10 score without chat format
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
#     --eval_pass_at_ks 10 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --save_dir results/codex_humaneval/tulu_7B_temp_0_8_nochat \
#     --model ../checkpoints/tulu_7B/ \
#     --tokenizer ../checkpoints/tulu_7B/ \
#     --use_vllm

# # Evaluating chatgpt using temperature 0.1 to get the pass@1 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --save_dir results/codex_humaneval/chatgpt_temp_0.1/ \
#     --eval_batch_size 10


# # Evaluating chatgpt using temperature 0.8 to get the pass@10 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --save_dir results/codex_humaneval/chatgpt_temp_0.8/ \
#     --eval_batch_size 10


# # Evaluating gpt4 using temperature 0.1 to get the pass@1 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --openai_engine "gpt-4-0314" \
#     --save_dir results/codex_humaneval/gpt4_temp_0.1 \
#     --eval_batch_size 1


# # Evaluating gpt4 using temperature 0.8 to get the pass@10 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --openai_engine "gpt-4-0314" \
#     --save_dir results/codex_humaneval/gpt4_temp_0.8 \
#     --eval_batch_size 1
