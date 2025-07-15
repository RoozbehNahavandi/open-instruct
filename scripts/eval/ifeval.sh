# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
# export CUDA_VISIBLE_DEVICES=0


# echo "Evaluating mdpo llama-3.1-tulu-3-8b-preference-mixture"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-7B-sft \
#     --model /fs/scratch/PAS2138/roozbehn99/open-instruct/models/mdpo/lm_Llama-3.1-Tulu-3-8B-SFT_llama-3.1-tulu-3-8b-preference-mixture_mdpo_nolora_2025-05-08_run211327 \
#     --tokenizer /fs/scratch/PAS2138/roozbehn99/open-instruct/models/mdpo/lm_Llama-3.1-Tulu-3-8B-SFT_llama-3.1-tulu-3-8b-preference-mixture_mdpo_nolora_2025-05-08_run211327 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# echo "Evaluating allenai/tulu-v2.5-ppo-13b-hh-rlhf-60k"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model allenai/tulu-v2.5-ppo-13b-hh-rlhf-60k \
#     --tokenizer allenai/tulu-v2.5-ppo-13b-hh-rlhf-60k \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# echo "Evaluating allenai/tulu-v2.5-ppo-13b-uf-mean"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model allenai/tulu-v2.5-ppo-13b-uf-mean \
#     --tokenizer allenai/tulu-v2.5-ppo-13b-uf-mean \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm



echo "Evaluating output/llama3_1b_finetuned"
python -m eval.ifeval.run_eval \
    --data_dir data/eval/ifeval/ \
    --save_dir results/ifeval/tulu-70B-dpo \
    --model output/llama3_1b_finetuned \
    --tokenizer output/llama3_1b_finetuned \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# echo "Evaluating Tulu-V2.5-7B_V1"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model models/base_ppo/Tulu-V2.5-7B_V1 \
#     --tokenizer models/base_ppo/Tulu-V2.5-7B_V1 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm
    
# echo "Evaluating allenai/tulu-v2.5-dpo-13b-uf-mean"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model allenai/tulu-v2.5-dpo-13b-uf-mean \
#     --tokenizer allenai/tulu-v2.5-dpo-13b-uf-mean \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm

# echo "Evaluating Tulu-2.5-7b-dpo"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model models/dpo/Tulu-2.5-7b-dpo \
#     --tokenizer models/dpo/Tulu-2.5-7b-dpo \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# echo "Evaluating lm_llama3_1b_finetuned_ultrafeedback_ppo_nolora_2025-05-05_run143334"
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model models/ppo/lm_llama3_1b_finetuned_ultrafeedback_ppo_nolora_2025-05-05_run143334 \
#     --tokenizer models/ppo/lm_llama3_1b_finetuned_ultrafeedback_ppo_nolora_2025-05-05_run143334 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# # Evaluating chatgpt
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0125" \
#     --eval_batch_size 20


# # Evaluating gpt4
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/gpt4-cot \
#     --openai_engine "gpt-4-0613" \
#     --eval_batch_size 20
