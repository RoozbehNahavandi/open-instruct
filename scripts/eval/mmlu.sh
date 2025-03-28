# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# # # Evaluating llama 7B model using 0 shot directly
# # python -m eval.mmlu.run_eval \
# #     --ntrain 0 \
# #     --data_dir data/eval/mmlu \
# #     --save_dir results/mmlu/llama-7B-0shot \
# #     --model_name_or_path ../hf_llama_models/7B \
# #     --tokenizer_name_or_path ../hf_llama_models/7B \
# #     --eval_batch_size 4 \
# #     --load_in_8bit


# # # Evaluating llama 7B model using 5 shot directly
# # python -m eval.mmlu.run_eval \
# #     --ntrain 5 \
# #     --data_dir data/eval/mmlu \
# #     --save_dir results/mmlu/llama-7B-5shot \
# #     --model_name_or_path ../hf_llama_models/7B \
# #     --tokenizer_name_or_path ../hf_llama_models/7B \
# #     --eval_batch_size 4 \
# #     --load_in_8bit


# # # Evaluating Tulu 7B model using 0 shot and chat format
# # python -m eval.mmlu.run_eval \
# #     --ntrain 0 \
# #     --data_dir data/eval/mmlu \
# #     --save_dir results/mmlu/tulu-7B-0shot \
# #     --model_name_or_path ../checkpoints/tulu_7B \
# #     --tokenizer_name_or_path ../checkpoints/tulu_7B \
# #     --eval_batch_size 4 \
# #     --load_in_8bit \
# #     --use_chat_format \
# #     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # # Evaluating Tulu 7B model using 5 shot and chat format
# # python -m eval.mmlu.run_eval \
# #     --ntrain 5 \
# #     --data_dir data/eval/mmlu \
# #     --save_dir results/mmlu/tulu-7B-5shot \
# #     --model_name_or_path ../checkpoints/tulu_7B \
# #     --tokenizer_name_or_path ../checkpoints/tulu_7B \
# #     --eval_batch_size 4 \
# #     --load_in_8bit \
# #     --use_chat_format \
# #     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model using 0-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B \
#     --tokenizer_name_or_path ../hf_llama2_models/7B \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# echo "Finished evaluation on llama2 0-shot"


# # Evaluating llama2 chat model using 5-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B \
#     --tokenizer_name_or_path ../hf_llama2_models/7B \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# echo "Finished evaluation on llama2 5-shot"


# # Evaluating llama2 model using 0-shot finetuned on tuluv2 dataset
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-0shot-checkpoint \
#     --model_name_or_path output/tulu_v2_7B \
#     --tokenizer_name_or_path output/tulu_v2_7B \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# echo "Finished evaluation on finetuned llama2 0-shot"

# # Evaluating llama2 model using 5-shot finetuned on tuluv2 dataset
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot-checkpoint \
#     --model_name_or_path output/tulu_v2_7B \
#     --tokenizer_name_or_path output/tulu_v2_7B \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# echo "Finished evaluation on finetuned llama2 5-shot"

# # Evaluating tulu_v2 model using 0-shot 
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-0shot-checkpoint \
#     --model_name_or_path allenai/tulu-2-7b \
#     --tokenizer_name_or_path allenai/tulu-2-7b \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# echo "Finished evaluation on allenai/tulu-2-7b 0-shot"

# # Evaluating tulu_v2 model using 5-shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot-checkpoint \
#     --model_name_or_path allenai/tulu-2-7b \
#     --tokenizer_name_or_path allenai/tulu-2-7b \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# echo "Finished evaluation on allenai/tulu-2-7b 5-shot"

# ____________________________________


# Evaluating dpo model using 5-shot
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama2-chat-7B-5shot-checkpoint \
    --model_name_or_path output/dpo_7b_recreate2\
    --tokenizer_name_or_path output/dpo_7b_recreate2 \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

echo "Finished evaluation on output/dpo_7b 5-shot"


# Evaluating dpo lora model using 0-shot 
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama2-chat-7B-0shot-checkpoint \
    --model_name_or_path output/dpo_7b_recreate2 \
    --tokenizer_name_or_path output/dpo_7b_recreate2 \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

echo "Finished evaluation on output/dpo_7b 0-shot"

# Evaluating dpo lora model using 5-shot
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama2-chat-7B-5shot-checkpoint \
    --model_name_or_path output/dpo_7b_lora_merged \
    --tokenizer_name_or_path output/dpo_7b_lora_merged \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

echo "Finished evaluation on output/dpo_7b_lora_merged"


# Evaluating dpo lora model using 0-shot 
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama2-chat-7B-0shot-checkpoint \
    --model_name_or_path output/dpo_7b_lora_merged \
    --tokenizer_name_or_path output/dpo_7b_lora_merged \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

echo "Finished evaluation on output/dpo_7b_lora_merged 0-shot"

# # Evaluating chatgpt using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating chatgpt using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating gpt4 using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # Evaluating gpt4 using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20