# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# # Evaluating llama 7B model using chain-of-thought
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/llama-7B-cot-4shot \
#     --model ../hf_llama2_models/7B \
#     --tokenizer ../hf_llama2_models/7B \
#     --n_shot 4 \
#     --use_vllm
# echo "Finished evaluation on llama2 (CoT)"


# # Evaluating llama 7B model using direct answering (no chain-of-thought)
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-no-cot-4shot \
#     --model ../hf_llama2_models/7B \
#     --tokenizer ../hf_llama2_models/7B \
#     --n_shot 4 \
#     --no_cot \
#     --use_vllm

# echo "Finished evaluation on llama2 (no CoT)"


# # Evaluating llama 7B finetuned model using chain-of-thought
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/llama-7B-cot-4shot \
#     --model output/tulu_v2_7B \
#     --tokenizer output/tulu_v2_7B \
#     --n_shot 4 \
#     --use_vllm

# echo "Finished evaluation on finetuned llama2 (CoT)"


# # Evaluating llama 7B finetuned model using direct answering (no chain-of-thought)
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-no-cot-4shot \
#     --model output/tulu_v2_7B \
#     --tokenizer output/tulu_v2_7B \
#     --n_shot 4 \
#     --no_cot \
#     --use_vllm

# echo "Finished evaluation on finetuned llama2 (no CoT)"


# # Evaluating allenai/tulu-2-7b model using chain-of-thought
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/llama-7B-cot-4shot \
#     --model allenai/tulu-2-7b \
#     --tokenizer allenai/tulu-2-7b \
#     --n_shot 4 \
#     --use_vllm

# echo "Finished evaluation on allenai/tulu-2-7b (CoT)"


# # Evaluating allenai/tulu-2-7b model using direct answering (no chain-of-thought)
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-no-cot-4shot \
#     --model allenai/tulu-2-7b \
#     --tokenizer allenai/tulu-2-7b \
#     --n_shot 4 \
#     --no_cot \
#     --use_vllm

# echo "Finished evaluation on allenai/tulu-2-7b (no CoT)"





# Evaluating output/dpo_7b_recreate2 model using chain-of-thought
python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/MATH/llama-7B-cot-4shot \
    --model output/dpo_7b_recreate2 \
    --tokenizer output/dpo_7b_recreate2 \
    --n_shot 4 \
    --use_vllm

echo "Finished evaluation on output/dpo_7b_recreate2 (CoT)"


# Evaluating output/dpo_7b_recreate2b model using direct answering (no chain-of-thought)
python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama-7B-no-cot-4shot \
    --model output/dpo_7b_recreate2 \
    --tokenizer output/dpo_7b_recreate2 \
    --n_shot 4 \
    --no_cot \
    --use_vllm

echo "Finished evaluation on output/dpo_7b_recreate2 (no CoT)"



# Evaluating output/dpo_7b_lora_merged model using chain-of-thought
python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/MATH/llama-7B-cot-4shot \
    --model output/dpo_7b_lora_merged \
    --tokenizer output/dpo_7b_lora_merged \
    --n_shot 4 \
    --use_vllm

echo "Finished evaluation on output/dpo_7b_lora_merged (CoT)"


# Evaluating output/dpo_7b_lora_merged model using direct answering (no chain-of-thought)
python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama-7B-no-cot-4shot \
    --model output/dpo_7b_lora_merged \
    --tokenizer output/dpo_7b_lora_merged \
    --n_shot 4 \
    --no_cot \
    --use_vllm

echo "Finished evaluation on output/dpo_7b_lora_merged (no CoT)"



# Evaluating output/dpo_7b_lora_merged2 model using chain-of-thought
python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/MATH/llama-7B-cot-4shot \
    --model output/dpo_7b_lora_merged2 \
    --tokenizer output/dpo_7b_lora_merged2 \
    --n_shot 4 \
    --use_vllm

echo "Finished evaluation on output/dpo_7b_lora_merged2 (CoT)"


# Evaluating output/dpo_7b_lora_merged2 model using direct answering (no chain-of-thought)
python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama-7B-no-cot-4shot \
    --model output/dpo_7b_lora_merged2 \
    --tokenizer output/dpo_7b_lora_merged2 \
    --n_shot 4 \
    --no_cot \
    --use_vllm

echo "Finished evaluation on output/dpo_7b_lora_merged2 (no CoT)"

# --------------------------------------------- #

# # Evaluating tulu 7B model using chain-of-thought and chat format
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/tulu-7B-cot-4shot \
#     --model ../checkpoints/tulu_7B \
#     --tokenizer ../checkpoints/tulu_7B \
#     --n_shot 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# # Evaluating llama2 chat model using chain-of-thought and chat format
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/llama2-chat-7B-cot-4shot \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --n_shot 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
#     --use_vllm


# # Evaluating chatgpt using chain-of-thought
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/chatgpt-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 4


# # Evaluating chatgpt using direct answering (no chain-of-thought)
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 4 \
#     --no_cot


# # Evaluating gpt4 using chain-of-thought
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/gpt4-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 4 


# # Evaluating gpt4 using direct answering (no chain-of-thought)
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/gpt4-no-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 4 \
#     --no_cot
