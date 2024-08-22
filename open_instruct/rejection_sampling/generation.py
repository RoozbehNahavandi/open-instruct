# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import copy
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Dict, List, Optional

from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

from open_instruct.rejection_sampling.api_generate import (  # Import your classes
    LLMGenerationConfig,
    LLMProcessor,
)

api = HfApi()
# we don't use `multiprocessing.cpu_count()` because typically we only have 12 CPUs
# and that the shards might be small
NUM_CPUS_FOR_DATASET_MAP = 4


@dataclass
class Args:
    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    save_filename: str = "completions.jsonl"
    skill: str = "chat"
    mode: str = "generation"  # Can be "generation" or "judgment"

    # upload config
    hf_repo_id: str = os.path.basename(__file__)[: -len(".py")]
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    add_timestamp: bool = True


@dataclass
class GenerationArgs:
    num_completions: int = 3
    temperature: float = 0.8
    response_length: int = 2048
    top_p: float = 0.9
    tensor_parallel_size: int = 1


@dataclass
class DatasetArgs:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"
    dataset_start_idx: int = 0
    dataset_end_idx: Optional[int] = 100
    sanity_check: bool = False
    sanity_check_size: int = 100


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


async def generate_with_openai(model_name: str, data_list: list, args: Args, gen_args: GenerationArgs):
    config = LLMGenerationConfig(model=model_name, num_completions=gen_args.num_completions)
    processor = LLMProcessor(config)
    results = await processor.process_batch(data_list, args, gen_args)
    return results


def generate_with_vllm(model_name_or_path: str, prompt_token_ids: List[int], gen_args: GenerationArgs):
    llm = LLM(model=model_name_or_path, tensor_parallel_size=gen_args.tensor_parallel_size)

    # filter out prompts which are beyond the model's max token length
    max_model_len = llm.llm_engine.scheduler_config.max_model_len
    prompt_token_ids_len = len(prompt_token_ids)
    prompt_token_ids = [item for item in prompt_token_ids if len(item) < max_model_len]
    if len(prompt_token_ids) != prompt_token_ids_len:
        print(f"Filtered out {prompt_token_ids_len - len(prompt_token_ids)} prompts which exceeds max token length")

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            n=gen_args.num_completions,
            temperature=gen_args.temperature,
            top_p=1.0,
            max_tokens=gen_args.response_length,
            include_stop_str_in_output=True,
        ),
    )

    return [
        {
            "outputs": [asdict(out) for out in output.outputs],
            "prompt": output.prompt,
            "prompt_logprobs": output.prompt_logprobs,
            "metrics": output.metrics,
        }
        for output in outputs
    ]


def format_conversation(messages: list) -> str:
    formatted_conversation = []

    # Iterate through the messages
    for message in messages:  # Exclude the last assistant message
        role = "User A" if message["role"] == "user" else "User B"
        content = message["content"].strip()
        formatted_conversation.append(f"{role}: {content}")

    # Join the conversation with a single newline
    return "\n".join(formatted_conversation)


def main(args: Args, dataset_args: DatasetArgs, gen_args: GenerationArgs):

    ds = load_dataset(dataset_args.dataset_name)
    if dataset_args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(min(dataset_args.sanity_check_size, len(ds[key]))))
    if dataset_args.dataset_end_idx is None:
        dataset_args.dataset_end_idx = len(ds[dataset_args.dataset_train_split])
    for key in ds:
        ds[key] = ds[key].select(range(dataset_args.dataset_start_idx, dataset_args.dataset_end_idx))
    pprint([dataset_args, args, gen_args])

    if "gpt-3.5" in args.model_name_or_path or "gpt-4" in args.model_name_or_path:
        ds = ds.map(
            lambda x: {"prompt": format_conversation(x["messages"][:-1])},
            num_proc=NUM_CPUS_FOR_DATASET_MAP,
        )
        messages = ds[dataset_args.dataset_train_split]["prompt"]
        responses = asyncio.run(generate_with_openai(args.model_name_or_path, messages, args, gen_args))
        outputs = [{"outputs": [{"text": response} for response in responses]}]

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        ds = ds.map(
            lambda x: {"prompt_token_ids": tokenizer.apply_chat_template(x["messages"][:-1])},
            num_proc=NUM_CPUS_FOR_DATASET_MAP,
        )
        prompt_token_ids = ds[dataset_args.dataset_train_split]["prompt_token_ids"]
        outputs = generate_with_vllm(args.model_name_or_path, prompt_token_ids, gen_args)

    # Assuming we generate n=3 completions per prompt; the outputs will look like:
    # prompt | completions
    # -------|------------
    # q1     | a1
    # q1     | a2
    # q1     | a3
    # q2     | a1
    # ...
    table = defaultdict(list)
    num_prompt_with_identical_completions = 0
    for output, messages in zip(outputs, ds[dataset_args.dataset_train_split]["messages"]):
        # if the model completions are exactly the same across all completions per prompt, we can skip this
        if len(set(tuple(item["text"]) for item in output["outputs"])) == 1:
            num_prompt_with_identical_completions += 1
            continue

        for item in output["outputs"]:
            new_messages = copy.deepcopy(messages[:-1])
            new_messages.append({"role": "assistant", "content": item["text"]})
            table["messages"].append(new_messages)
            table["model_completion"].append(item["text"])
            table["reference_completion"].append(messages[-1]["content"])

    print(f"Number prompts with identical completions: {num_prompt_with_identical_completions}")
    save_jsonl(args.save_filename, table)

    if args.push_to_hub:
        if args.hf_entity is None:
            args.hf_entity = api.whoami()["name"]
        full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        timestamp = f"_{int(time.time())}"
        if args.add_timestamp:
            full_repo_id += timestamp
        api.create_repo(full_repo_id, repo_type="dataset", exist_ok=True)
        for f in [__file__, args.save_filename]:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f.split("/")[-1],
                repo_id=full_repo_id,
                repo_type="dataset",
            )
        repo_full_url = f"https://huggingface.co/datasets/{full_repo_id}"
        print(f"Pushed to {repo_full_url}")
        run_command = " ".join(["python"] + sys.argv)
        sft_card = RepoCard(
            content=f"""\
# allenai/open_instruct: Generation Dataset

See https://github.com/allenai/open-instruct/blob/main/docs/algorithms/rejection_sampling.md for more detail

## Configs

```
args:
{pformat(vars(args))}

dataset_args:
{pformat(vars(dataset_args))}

gen_args:
{pformat(vars(gen_args))}
```

## Reproduce this dataset

1. Download the `{[f.split("/")[-1] for f in [__file__, args.save_filename]]}` from the {repo_full_url}.
2. Run `{run_command}`
"""
        )
        sft_card.push_to_hub(
            full_repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    parser = HfArgumentParser((Args, DatasetArgs, GenerationArgs))
    args, dataset_args, gen_args = parser.parse_args_into_dataclasses()
    main(args, dataset_args, gen_args)
