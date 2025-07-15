import gc
import json
import os
import random
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import List, Literal, Optional, Tuple, Dict, Union
import subprocess
import torch.distributed as dist
import socket
import dataclasses


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import DatasetDict
from huggingface_hub import HfApi
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from vllm import LLM, SamplingParams

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTDatasetProcessor,
    SimpleGenerateCollator,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    save_with_accelerate,
    truncate_response,
    unwrap_model_for_generation,
)
from open_instruct.utils import (
    ArgumentParserPlus,
    combine_dataset,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)
from open_instruct.vllm_utils import vllm_single_gpu_patch
from eval.ifeval import instructions_registry

api = HfApi()
INVALID_LOGPROB = 1.0

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]



@dataclass
class Args:
    # required dataset args
    dataset_mixer: str = None
    """A dictionary of datasets (local or HF) to sample from."""
    dataset_train_splits: List[str] = None
    """The dataset splits to use for training"""
    dataset_eval_mixer: Optional[str] = None
    """A dictionary of datasets (local or HF) to sample from for evaluation"""
    dataset_eval_splits: Optional[List[str]] = None
    """The dataset splits to use for evaluation"""
    dataset_mixer_dict: Optional[dict] = None
    """The dataset mixer as a dictionary"""
    dataset_eval_mixer_dict: Optional[dict] = None
    """The dataset eval mixer as a dictionary"""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""
    local_dataloader_batch_size: Optional[int] = None
    """The batch size per GPU for the dataloader"""

    # online settings
    num_epochs: int = 4
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    # reward_model_path: str = "EleutherAI/pythia-160m"
    # """the path to the reward model 1"""
    # reward_model_revision: Optional[str] = None
    # """the revision of the reward model 1"""
    reward_model_paths: List[str] = field(default_factory=lambda: ["EleutherAI/pythia-160m"])
    """List of HuggingFace paths for reward models (e.g., helpfulness, safety, etc.)"""
    reward_model_revisions: Optional[List[Optional[str]]] = None
    """Optional list of git revisions for each reward model"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model 1"""
    reward_model2_path: str = None
    """the path to the reward model 2"""
    reward_model_revision: Optional[str] = None
    """the revision of the reward model 1"""
    reward_model2_revision: Optional[str] = None
    """the revision of the reward model 2"""
    # generation config
    response_length: int = 53
    """the length of the response"""
    stop_token: Optional[Literal["eos", "period"]] = None
    """the stop token"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = True
    """whether to penalize responses that do not contain `stop_token_id`"""

    # online PPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    cliprange: float = 0.2
    """the clip range"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    kl_estimator: Literal["kl1", "kl2", "kl3"] = "kl1"

    # vLLM settings. NOTE: currently we need to place the vLLM model on a separate GPU
    # for generation to work properly because vLLM would pre-alocate the memory.
    # To do so, we would need to do a moneky patch `vllm_single_gpu_patch` to make sure
    # the vLLM model is placed on the correct GPU.
    vllm_device: str = "cuda:1"
    """the device placement of the vllm model; typically we place the vllm model on a decicated GPU"""
    vllm_gpu_memory_utilization: float = 0.8
    """the GPU memory utilization of the vllm model; passed to `gpu_memory_utilization` to the `vLLM` instance"""
    # async setting
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""
 
    # Eval RM settings
    evalrm_name_or_path: str = None
    """The name or path of the evaluation reward model"""
    ifeval: bool = True
    """if to use IF eval"""
    ifeval_eval_freq: int = 2

    # wandb and HF tracking configs
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_ppo"
    """The wandb's project name"""
    wandb_run_name: str = "ppo_tuluv2_7b"
    """The wandb run's name"""
    wandb_entity: Optional[str] = 'roozbeh-n99'
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""
    checkpoint_output_dir: Optional[str] = None
    """Where to save the model checkpoints in case of preemption"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        self.dataset_mixer_dict, self.dataset_mixer = process_dataset_mixer(self.dataset_mixer)
        if self.dataset_eval_mixer is not None:
            self.dataset_eval_mixer_dict, self.dataset_eval_mixer = process_dataset_mixer(self.dataset_eval_mixer)


def process_dataset_mixer(value) -> Tuple[Optional[dict], Optional[str]]:
    # if passed through cli: convert the dataset mixers to dictionaries
    if isinstance(value, str):
        return json.loads(value), value
    # if passed through yaml: convert the dataset mixers to strings
    elif isinstance(value, dict):
        return value, json.dumps(value)
    else:
        raise ValueError("Input must be either a string or a dictionary")


def calculate_runtime_args_and_accelerator(args: Args, model_config: ModelConfig) -> Accelerator:
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)


    args.world_size = accelerator.num_processes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    # set a unique run name with the current timestamp
    time_int = broadcast(time_tensor, 0).item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    args.mini_batch_size = exact_div(
        args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
    )
    args.local_mini_batch_size = exact_div(
        args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
    )
    args.num_training_steps = args.total_episodes // args.batch_size
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    # PPO logic: do checks and set up dataloader batch size
    if args.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    args.local_dataloader_batch_size = args.local_batch_size
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.with_tracking and accelerator.is_main_process:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
    return accelerator

def vllm_generate(
    model_name_or_path: str,
    model_revision: Optional[str],
    max_model_len: int,
    vllm_device: str,
    vllm_gpu_memory_utilization: float,
    generation_config: SamplingParams,
    response_ids_Q: Queue,
    param_prompt_Q: Queue,
    num_training_steps: int,
    sample_evaluation_prompt_token_ids: Optional[List[int]],
    evaluation_Q: Queue,
    eval_freq: int,
    resume_training_step: int,
    tokenizer: AutoTokenizer = None,
):
    vllm_single_gpu_patch()
    llm = LLM(
        model=model_name_or_path,
        revision=model_revision,
        tokenizer_revision=model_revision,
        tensor_parallel_size=1,
        device=vllm_device,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    print("🔥🔥🔥 vllm loaded")
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    for training_step in range(resume_training_step, num_training_steps + 1):
        items = param_prompt_Q.get()
        if items is None:
            break
        unwrapped_model, g_queries_list = items
        if unwrapped_model is not None:
            start_time = time.time()
            llmp.load_weights(unwrapped_model.named_parameters())
            print(
                f"🔥🔥🔥 Loading weights using shared memory; Time to load weights: {time.time() - start_time:.2f} seconds"
            )
        generation_start_time = time.time()
        # print(f'decoded prompt in vllm_generate: {tokenizer.batch_decode(g_queries_list, skip_special_tokens=False)}')
        outputs = llm.generate(prompt_token_ids=g_queries_list, sampling_params=generation_config)
        response_ids = [list(output.outputs[0].token_ids) for output in outputs]
        # print(f'decoded output in vllm_generate: {tokenizer.batch_decode(response_ids, skip_special_tokens=False)}')
        print(f"🔥🔥🔥 Generation time: {time.time() - generation_start_time:.2f} seconds")
        response_ids_Q.put(response_ids)
        print(f'before eval - training step: {training_step}, {(training_step - 1) % eval_freq}')
        if sample_evaluation_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
            print(f'🔥🔥🔥 Running evaluation at step {training_step}')
            prompt_token_ids = sample_evaluation_prompt_token_ids.tolist()


            eval_config = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=2048,
                include_stop_str_in_output=True,
            )
            eval_outputs = llm.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=eval_config
            )
            # 🔍 Print first prompt and response for debugging
            first_prompt_ids = prompt_token_ids[:1]
            first_prompt = tokenizer.decode(first_prompt_ids[0], skip_special_tokens=False)
            first_output_ids = eval_outputs[0].outputs[0].token_ids
            first_output = tokenizer.decode(first_output_ids, skip_special_tokens=False)

            print("🧪 [PPO] First eval prompt:\n", first_prompt)
            print("🧪 [PPO] First eval output:\n", first_output)
            
            response_ids = [list(output.outputs[0].token_ids) for output in eval_outputs]
            evaluation_Q.put(response_ids)

def send_queries(accelerator, unwrapped_model, tokenizer, param_prompt_Q, queries):
    g_queries_list = gather_object(queries.tolist())
    if accelerator.is_main_process:
        g_queries_list = [
            [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id] for item in g_queries_list
        ]  # remove padding
        param_prompt_Q.put((unwrapped_model, g_queries_list))

def read_prompt_list(input_jsonl_filename):
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            inputs.append(
                InputExample(key=example["key"],
                            instruction_id_list=example["instruction_id_list"],
                            prompt=example["prompt"],
                            kwargs=example["kwargs"]))
    return inputs


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instrutions are followed."""
    # response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    # response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )

# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(torch.nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits

    def gradient_checkpointing_enable(self):
        self.policy.gradient_checkpointing_enable()
        self.value_model.gradient_checkpointing_enable()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened






def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):

    # init_dist(args)
    accelerator = calculate_runtime_args_and_accelerator(args, model_config)
    local_seed = args.seed + accelerator.process_index
    subprocess.run(['nvidia-smi'], shell=True)


    # set up experiment tracking and seeds
    all_configs = {}
    if is_beaker_job():
        args.checkpoint_output_dir = os.environ.get("CHECKPOINT_OUTPUT_DIR", args.output_dir)
        beaker_config = maybe_get_beaker_config()
        # try saving to the beaker `/output`, which will be uploaded to the beaker dataset
        if len(beaker_config.beaker_dataset_id_urls) > 0:
            args.output_dir = "/output"
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(dataset_config), **asdict(model_config))
    if accelerator.is_main_process:
        if args.with_tracking:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                sync_tensorboard=True,
                config=all_configs,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    device = torch.device(f"cuda:{accelerator.local_process_index}")
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    # create a tokenizer (pad from right)
    config = AutoConfig.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision, padding_side="right"
    )
    if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
        tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]

    # create the dataset
    dataset_dict = DatasetDict()
    dataset_processor = SFTDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    train_dataset = combine_dataset(
        args.dataset_mixer_dict,
        splits=args.dataset_train_splits,
        columns_to_keep=['chosen'],
    )
    train_dataset = train_dataset.map(
        lambda row: {"prompt": row["chosen"][-2]["content"] if row["chosen"][-2]["role"] == "user" else ""},
        desc="Extracting prompt from last user message"
    )
    train_dataset = train_dataset.remove_columns(["chosen"])
    # print len of train_dataset
    # print("🔥🔥🔥 Train dataset length:", len(train_dataset))
    # print first sample of train_dataset train_dataset[0]
    print(f'11 train_dataset[0]: {train_dataset[0]}')  
 

    if dataset_config.sanity_check:
        train_dataset = train_dataset.select(
            range(0, min(len(train_dataset), dataset_config.sanity_check_max_samples))
        )
    with accelerator.main_process_first():
        train_dataset = dataset_processor.tokenize(train_dataset)
        # print(f'ananan train dataset[1]: {train_dataset[1]}')  # print second sample of train_dataset
        train_dataset = dataset_processor.filter(train_dataset)
        # print(f'ananan train dataset[2]: {train_dataset[2]}')  # print third sample of train_dataset

    # # Print a single sample to verify tokenization + chat template
    # sample = train_dataset[0]
    # print("\n🧾 Raw prompt string:", sample["prompt"])
    # print("🧱 input_ids_prompt:", sample[INPUT_IDS_PROMPT_KEY])
    # print("🧾 Decoded prompt (no skip):", tokenizer.decode(sample[INPUT_IDS_PROMPT_KEY], skip_special_tokens=False))
    # print("🧾 Decoded prompt (skip specials):", tokenizer.decode(sample[INPUT_IDS_PROMPT_KEY], skip_special_tokens=True))

    dataset_dict["train"] = train_dataset
    eval_dataset = None
    if args.dataset_eval_mixer is not None:
        eval_dataset = combine_dataset(
            args.dataset_eval_mixer_dict,
            splits=args.dataset_eval_splits,
            columns_to_keep=['chosen'],
        )
        eval_dataset = eval_dataset.map(
            lambda row: {"prompt": row["chosen"][-2]["content"] if row["chosen"][-2]["role"] == "user" else ""},
            desc="Extracting prompt from last user message"
        )
        eval_dataset = eval_dataset.remove_columns(["chosen"])
        eval_dataset = eval_dataset.select(range(0, min(len(eval_dataset), dataset_config.sanity_check_max_samples)))
        with accelerator.main_process_first():
            eval_dataset = dataset_processor.tokenize(eval_dataset)
            eval_dataset = dataset_processor.filter(eval_dataset)
        dataset_dict["eval"] = eval_dataset

    # some more runtime logging
    if accelerator.is_main_process:
        pprint([args, dataset_config, model_config])
        # visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
        if args.with_tracking:
            # upload the visualized token length
            dataset_processor.get_token_length_visualization(
                dataset_dict, save_path=f"runs/{args.run_name}/token_length.png"
            )
            wandb.log({"token_length": wandb.Image(f"runs/{args.run_name}/token_length.png")})

    # create the model and optimizer
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    value_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        revision=args.reward_model_revision,
        num_labels=1,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        revision=args.reward_model_revision,
        num_labels=1,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    # if policy.config.vocab_size != reward_model.config.vocab_size:
    #     raise ValueError(
    #         "Policy and reward model must have the same vocab size. "
    #         f"Policy: {policy.config.vocab_size}, Reward: {reward_model.config.vocab_size}. "
    #         "If they don't have the same vocab size, the policy could generate tokens which "
    #         "is going to cause index out of bound error in the reward model."
    #     )
    model = PolicyAndValueWrapper(policy, value_model)
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    for module in [model, ref_model, reward_model]:
        disable_dropout_in_model(module)
    if args.stop_token:
        if args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        if args.stop_token == "period":
            args.stop_token_id = tokenizer.encode(".")[0]
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_training_steps * args.num_train_epochs,
    )

    data_collator = SimpleGenerateCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.local_dataloader_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,  # needed; otherwise the last batch will be of ragged shape
    )
    print(f'len(train_dataset): {len(train_dataset)}')  # print length of train_dataset
    # data_iter = iter(dataloader)
    # data = next(data_iter)
    # print(f'dataaa: {data}')  # print first sample of dataloader
    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    torch.manual_seed(local_seed)

    # resume from preemption
    resume_training_step = 1
    if os.path.exists(args.checkpoint_output_dir):
        for item in os.listdir(args.checkpoint_output_dir):
            if "step_" in item:
                old_checkpoint_path = os.path.join(args.checkpoint_output_dir, item)
                # check if the directory is empty
                if len(os.listdir(old_checkpoint_path)) == 0:
                    continue
                accelerator.load_state(old_checkpoint_path)
                resume_training_step = int(item.split("_")[-1])
                print("Resuming training from step", resume_training_step)
                if accelerator.is_main_process:
                    shutil.rmtree(old_checkpoint_path)
                break
    resume_training_step > 1

    # handle preemption
    class PreemptionHandler:
        preemptied = False

        def __init__(self):
            signal.signal(signal.SIGTERM, self.exit_gracefully)

        def exit_gracefully(self, signum, frame):
            output_dir = os.path.join(args.checkpoint_output_dir, f"step_{training_step - 1}")
            print(f"SIGTERM received, saving to {output_dir} from {accelerator.local_process_index}")
            accelerator.save_state(output_dir)
            if accelerator.is_main_process and args.with_tracking:
                wandb.log({"preempted": True}, commit=True)
                wandb.mark_preempting()
            if accelerator.is_main_process:
                try:
                    param_prompt_Q.put(None, timeout=20)
                    response_ids_Q.get(timeout=20)
                    print("vllm thread terminated")
                except Exception as e:
                    print(e)
            self.preemptied = True

        

    ph = PreemptionHandler()

    # deepspeed setup
    is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    mixed_precision = accelerator.state.mixed_precision
    if is_deepspeed_enabled:
        reward_model = prepare_deepspeed(reward_model, args.per_device_train_batch_size, mixed_precision)
        ref_model = prepare_deepspeed(ref_model, args.per_device_train_batch_size, mixed_precision)
    else:
        reward_model = reward_model.to(device)
        ref_model = ref_model.to(device)

    # online generation config
    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
    )
    param_prompt_Q = None
    response_ids_Q = None
    evaluation_Q = None
    if accelerator.is_main_process:
        response_ids_Q = Queue(maxsize=1)
        param_prompt_Q = Queue(maxsize=1)
        evaluation_Q = Queue(maxsize=1)
        LOCAL_NUM_EVAL_SAMPLES = 4
        num_eval_samples = LOCAL_NUM_EVAL_SAMPLES * accelerator.num_processes
        sample_evaluation_prompt_token_ids = None
        # if eval_dataset is not None:
        #     sample_evaluation_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]
        if args.ifeval:
            ifeval_inputs = read_prompt_list("data/eval/ifeval/input_data.jsonl")
            print("🔍 Number of ifeval prompts:", len(ifeval_inputs))
            print("📌 First raw prompt:")
            print(ifeval_inputs[0].prompt)

            # sample_evaluation_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]
            sample_evaluation_prompt_token_ids = tokenizer(
                [inp.prompt for inp in ifeval_inputs], return_tensors="pt", padding=True, truncation=True
            )["input_ids"]
            # Print tokenized version of first prompt
            print("🧩 First prompt token IDs:")
            print(sample_evaluation_prompt_token_ids[0].tolist())

            # Decode to double check
            print("🔁 First prompt decoded back:")
            print(tokenizer.decode(sample_evaluation_prompt_token_ids[0], skip_special_tokens=True))
        thread = threading.Thread(
            target=vllm_generate,
            args=(
                model_config.model_name_or_path,
                model_config.model_revision,
                dataset_config.max_prompt_token_length + args.response_length,
                args.vllm_device,
                args.vllm_gpu_memory_utilization,
                generation_config,
                response_ids_Q,
                param_prompt_Q,
                args.num_training_steps,
                sample_evaluation_prompt_token_ids,
                evaluation_Q,
                args.ifeval_eval_freq,
                resume_training_step,
                tokenizer
            ),
        )
        thread.start()
    torch.cuda.set_device(device)

    g_vllm_responses = torch.zeros((args.batch_size, args.response_length), device=device, dtype=torch.long)

    # set up the metrics and initial states
    stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    vf_loss_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    local_metrics = torch.zeros((20,), device=device)
    episode = args.batch_size * (resume_training_step - 1)
    model.train()

    # training loop
    start_time = time.time()
    data = next(iter_dataloader)
    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)


    send_queries(accelerator, None, tokenizer, param_prompt_Q, queries_next)

    for _ in range(1, resume_training_step):  # we didn't store scheduler state
        scheduler.step()

    for training_step in range(resume_training_step, args.num_training_steps + 1):
        # print(f"[Rank {accelerator.process_index}] Hostname: {socket.gethostname()}, Device: {torch.cuda.current_device()}, GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        episode += args.batch_size
        scheduler.step()
        queries = queries_next
        if ph.preemptied:
            break

        if accelerator.is_main_process:
            try:
                evaluation_response_ids = evaluation_Q.get_nowait()
            except Empty:
                print("🙈 No evaluation outputs at this step. Skipping eval.")
            else:
                # Proceed with decoding and eval only if there's something
                prompts = tokenizer.batch_decode(sample_evaluation_prompt_token_ids, skip_special_tokens=True)
                completions = tokenizer.batch_decode(evaluation_response_ids, skip_special_tokens=True)
                completions = [c.replace(tokenizer.pad_token, "") for c in completions]

                strict_outputs = [
                    test_instruction_following_strict(inp, completions[i])
                    for i, inp in enumerate(ifeval_inputs)
                ]
                loose_outputs = [
                    test_instruction_following_loose(inp, completions[i])
                    for i, inp in enumerate(ifeval_inputs)
                ]

                acc_strict = sum(o.follow_all_instructions for o in strict_outputs) / len(strict_outputs)
                acc_loose = sum(o.follow_all_instructions for o in loose_outputs) / len(loose_outputs)

                print("🔥🔥🔥 Evaluation responses received")
                print(f"📊 IFEVAL strict accuracy: {acc_strict:.4f}")
                print(f"📊 IFEVAL loose  accuracy: {acc_loose:.4f}")

                table = {
                    "prompt": prompts,
                    "response": completions,
                }
                df = pd.DataFrame(table)

                wandb.log({
                    "sample_completions": wandb.Table(dataframe=df),
                    "ifeval/acc_strict": acc_strict,
                    "ifeval/acc_loose": acc_loose,
                })
                print_rich_table(df.head(5))

            # Print the keys in the batch

        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            if unwrapped_model is None:
                continue  # skip this loop on non-main ranks

            generation_model = unwrapped_model.policy
            if args.async_mode:
                if training_step != 1:
                    data = next(iter_dataloader)
                    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
                send_queries(accelerator, generation_model, tokenizer, param_prompt_Q, queries_next)
            else:
                if training_step != 1:
                    # NOTE: important: the indent here is different for sync mode
                    # we also set to use `queries = queries_next` immediately
                    data = next(iter_dataloader)
                    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
                    send_queries(accelerator, generation_model, tokenizer, param_prompt_Q, queries_next)
                    queries = queries_next

            # Print the keys in the batch

            # # Decode a few input prompts
            # if INPUT_IDS_PROMPT_KEY in data:
            #     input_ids = data[INPUT_IDS_PROMPT_KEY][:3]  # take first 3
            #     decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
  
            # print("Prompt text:", tokenizer.batch_decode(queries_next, skip_special_tokens=False))
            training_time_start = time.time()
            with torch.no_grad():
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                if accelerator.is_main_process:
                    g_response_token_ids = response_ids_Q.get()
                    DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
                    g_padded_response_ids = [
                        response + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                        for response in g_response_token_ids
                    ]
                    # Decode each response
                    decoded_responses = tokenizer.batch_decode(g_response_token_ids, skip_special_tokens=False)
                    # for i, resp in enumerate(decoded_responses):
                    #     print(f"Decoded response [{i}]: {repr(resp)}")
                    for item in g_padded_response_ids:
                        assert len(item) == args.response_length
                        for inner_item in item:
                            if not inner_item < config.vocab_size:
                                assert inner_item < config.vocab_size, f"{inner_item=}, {tokenizer.vocab_size=}"
                    g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                    g_vllm_responses[:] = g_padded_response_ids
                broadcast(g_vllm_responses, 0)
                local_vllm_responses = g_vllm_responses[
                    accelerator.local_process_index
                    * queries.shape[0] : (accelerator.local_process_index + 1)
                    * queries.shape[0]
                ]
                query_responses = torch.cat((queries, local_vllm_responses), 1)
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    output = forward(generation_model, query_response, tokenizer.pad_token_id)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.temperature + 1e-7
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del output, logits, all_logprob
                    torch.cuda.empty_cache()

                    # Print the first sample in the mini-batch
                    idx = 0  # you can change this to print others in the batch


 
                    ref_output = forward(ref_model, query_response, tokenizer.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                        postproc = postprocessed_response[idx]


                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)

                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    reward_outputs = {}
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )

                    print(f"🟪 score: {score} FULL SEQUENCE AFTER POSTPROCESS:{repr(tokenizer.decode(postprocessed_query_response[0], skip_special_tokens=False))}")

                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, tokenizer.pad_token_id, context_length
                    )

                    # print(f'score: {score} {score.shape}, full_value: {full_value} {full_value.shape}')
                    # print('--------------')
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                global_scores = accelerator.gather(scores)
                accelerator.print(f"global_scores: {global_scores}, {global_scores.mean()}")
                values = torch.cat(values, 0)
                print(f'responses.shape: {responses.shape}, postprocessed_responses.shape: {postprocessed_responses.shape}')
                print(f'values.shape: {values.shape}')
                del (logprob, ref_logprob, full_value, value, score)
                gc.collect()
                torch.cuda.empty_cache()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_stop_token = torch.any(postprocessed_responses == args.stop_token_id, dim=-1)
                # NOTE: only apply the stop token filter if the response is long enough
                # otherwise the model could learn to generate the first token as the stop token
                contain_stop_token = contain_stop_token & (sequence_lengths >= args.min_response_length)
                if args.non_stop_penalty:
                    scores = torch.where(
                        contain_stop_token, scores, torch.full_like(scores, args.penalty_reward_value)
                    )

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl1 = logprobs - ref_logprobs
                kl2 = (kl1) ** 2 / 2
                kl3 = (-kl1).exp() - 1 + kl1
                if args.kl_estimator == "kl1":
                    kl = kl1
                elif args.kl_estimator == "kl2":
                    kl = kl2
                elif args.kl_estimator == "kl3":
                    kl = kl3
                # print(f"{accelerator.local_process_index=}, {kl.sum(1)=}")
                non_score_reward = -args.beta * kl
                non_score_reward_sum = non_score_reward.sum(1)
                rlhf_reward = scores + non_score_reward_sum
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)
                print(f'rewards.shape: {rewards.shape}')

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                print(f'advantages.shape: {advantages.shape}, returns.shape: {returns.shape}, ')

                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        for epoch_idx in range(args.num_epochs):
            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                    with accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        mb_return = returns[micro_batch_inds]
                        mb_values = values[micro_batch_inds]

                        output, vpred_temp = forward(model, mb_query_responses, tokenizer.pad_token_id)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.cliprange_value,
                            mb_values + args.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                        loss = pg_loss + args.vf_coef * vf_loss

                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropy_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()

                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                # fmt: off
                del (
                    output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                    vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                    pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                    mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                # del everything and empty cache
                torch.cuda.empty_cache()
        with torch.no_grad():
            local_metrics[0] = sequence_lengths.float().mean()
            local_metrics[1] = (responses == args.stop_token_id).sum().float().mean()
            local_metrics[2] = kl.sum(1).mean()
            local_metrics[3] = (-logprobs).sum(1).mean()
            local_metrics[4] = non_score_reward_sum.mean()
            local_metrics[5] = rlhf_reward.mean()
            local_metrics[6] = scores.mean()
            local_metrics[7] = approxkl_stats.mean()
            local_metrics[8] = pg_clipfrac_stats.mean()
            local_metrics[9] = pg_loss_stats.mean()
            local_metrics[10] = vf_loss_stats.mean()
            local_metrics[11] = vf_clipfrac_stats.mean()
            local_metrics[12] = entropy_stats.mean()
            local_metrics[13] = ratio_stats.mean()
            local_metrics[14] = ratio_stats.var()
            local_metrics[15] = ((kl) ** 2 / 2).sum(1).mean()
            local_metrics[16] = ((-kl).exp() - 1 + kl).sum(1).mean()
            global_metrics = accelerator.reduce(local_metrics, reduction="mean").tolist()
            metrics = {
                "episode": episode,
                "training_step": training_step,
                "lr": scheduler.get_last_lr()[0],
                "epoch": episode / len(train_dataset),
                "time/from_scratch": time.time() - start_time,
                "time/training": time.time() - training_time_start,
                "val/sequence_lengths": global_metrics[0],
                "val/num_stop_token_ids": global_metrics[1],
                "objective/kl": global_metrics[2],
                "objective/kl2": global_metrics[15],
                "ojbective/kl3": global_metrics[16],
                "objective/entropy": global_metrics[3],
                "objective/non_score_reward": global_metrics[4],
                "objective/rlhf_reward": global_metrics[5],
                "objective/scores": global_metrics[6],
                "policy/approxkl_avg": global_metrics[7],
                "policy/clipfrac_avg": global_metrics[8],
                "loss/policy_avg": global_metrics[9],
                "loss/value_avg": global_metrics[10],
                "val/clipfrac_avg": global_metrics[11],
                "policy/entropy_avg": global_metrics[12],
                "val/ratio": global_metrics[13],
                "val/ratio_var": global_metrics[14],
            }
            if accelerator.is_main_process:
                print_rich_single_line_metrics(metrics)
                for key, value in metrics.items():
                    writer.add_scalar(key, value, episode)
        del (queries, responses, postprocessed_responses, logprobs, ref_logprobs, sequence_lengths, scores)
        del (metrics, kl, non_score_reward, rlhf_reward)
        gc.collect()
        torch.cuda.empty_cache()
        # if training_step % 5 == 0:
        #     print(f'training_step: {training_step}')
        # if not training_step % 10:
        #     output_dir = f"step_{training_step}"
        #     if args.checkpoint_output_dir is not None:
        #         output_dir = os.path.join(args.checkpoint_output_dir, output_dir)
        #     accelerator.save_state(output_dir)

    if not ph.preemptied:
        # save model
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        original_tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, revision=model_config.model_revision
        )
        print(f'Saving the model to {args.output_dir}')
        save_with_accelerate(
            accelerator,
            model,
            original_tokenizer,
            args.output_dir,
            model_attribute_to_save="policy",
        )

        # # Ai2 specific logic
        # if is_beaker_job() and accelerator.is_main_process:
        #     if args.hf_metadata_dataset:
        #         dataset_list = list(args.dataset_mixer_dict.keys())
        #         # mainly just focussing here on what would be useful for the leaderboard.
        #         # wandb will have even more useful information.
        #         metadata_blob = {
        #             "model_name": args.exp_name,
        #             "model_type": "sft",
        #             "datasets": dataset_list,
        #             "base_model": model_config.model_name_or_path,
        #             "wandb_path": wandb.run.get_url(),
        #             "beaker_experiment": beaker_config.beaker_experiment_url,
        #             "beaker_datasets": beaker_config.beaker_dataset_id_urls,
        #         }
        #         upload_metadata_to_hf(
        #             metadata_blob,
        #             "metadata.json",
        #             args.hf_metadata_dataset,
        #             "results/" + args.hf_repo_revision,  # to match what the auto-evals name as.
        #         )

        #     if args.try_launch_beaker_eval_jobs and len(beaker_config.beaker_dataset_id_urls) > 0:
        #         command = f"""\
        #         python mason.py  \
        #             --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/s2-cirrascale ai2/general-cirrascale \
        #             --priority low \
        #             --preemptible \
        #             --budget ai2/allennlp \
        #             --workspace ai2/tulu-2-improvements \
        #             --image nathanl/open_instruct_auto \
        #             --pure_docker_mode \
        #             --gpus 0 -- python scripts/wait_beaker_dataset_model_upload_then_evaluate_model.py \
        #             --beaker_workload_id {beaker_config.beaker_workload_id} \
        #             --model_name {args.hf_repo_revision}
        #         """
        #         process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #         stdout, stderr = process.communicate()
        #         print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
        #         print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
        #         print(f"Submit jobs after model training is finished - process return code: {process.returncode}")

        if args.push_to_hub:
            push_folder_to_hub(
                accelerator,
                args.output_dir,
                args.hf_repo_id,
                args.hf_repo_revision,
            )
        # if accelerator.is_main_process:
        #     # remove args.checkpoint_output_dir
        #     if os.path.exists(args.checkpoint_output_dir):
        #         shutil.rmtree(args.checkpoint_output_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())