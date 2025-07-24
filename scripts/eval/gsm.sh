# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.


echo "Evaluating output/llama3_1b_finetuned"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama-7B-cot-8shot \
    --model output/llama3_1b_finetuned \
    --tokenizer output/llama3_1b_finetuned \
    --n_shot 8 \
    --use_vllm


# echo "Evaluating Tulu-V2.5-7B_V1"
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-cot-8shot \
#     --model models/base_ppo/Tulu-V2.5-7B_V1 \
#     --tokenizer models/base_ppo/Tulu-V2.5-7B_V1 \
#     --n_shot 8 \
#     --use_vllm


# echo "Evaluating allenai/tulu-v2.5-dpo-13b-uf-mean"
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-cot-8shot \
#     --model allenai/tulu-v2.5-dpo-13b-uf-mean \
#     --tokenizer allenai/tulu-v2.5-dpo-13b-uf-mean \
#     --n_shot 8 \
#     --use_vllm

# echo "Evaluating meta-llama/Llama-2-13b-hf"
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-cot-8shot \
#     --model meta-llama/Llama-2-13b-hf \
#     --tokenizer meta-llama/Llama-2-13b-hf \
#     --n_shot 8 \
#     --use_vllm


# echo "Evaluating mdpo llama-3.1-tulu-3-8b-preference-mixture"
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-no-cot-8shot \
#     --model models/mdpo/lm_Llama-3.1-Tulu-3-8B-SFT_llama-3.1-tulu-3-8b-preference-mixture_mdpo_nolora_2025-05-08_run211327 \
#     --tokenizer models/mdpo/lm_Llama-3.1-Tulu-3-8B-SFT_llama-3.1-tulu-3-8b-preference-mixture_mdpo_nolora_2025-05-08_run211327 \
#     --n_shot 8 \
#     --no_cot \
#     --use_vllm


# echo "Evaluating llama 3 1b mdpo"
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/tulu-7B-cot-8shot \
#     --model models/mdpo/lm_llama3_1b_finetuned_ultrafeedback_mdpo_nolora_2025-05-05_run120145 \
#     --tokenizer models/mdpo/lm_llama3_1b_finetuned_ultrafeedback_mdpo_nolora_2025-05-05_run120145 \
#     --n_shot 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# # Evaluating llama2 chat model using chain-of-thought and chat format
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama2-chat-7B-cot-8shot \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --n_shot 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
#     --use_vllm


# # Evaluating chatgpt using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # Evaluating chatgpt using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot


# # Evaluating gpt4 using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/gpt4-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # Evaluating gpt4 using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/gpt4-no-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot
