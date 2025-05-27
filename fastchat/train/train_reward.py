# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=500 \
    --max_length=512 \
"""
import pathlib
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

import transformers
from fastchat.train.reward_trainer import RewardTrainer
from dataclasses import dataclass, field
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict, Optional, Sequence
import math
from functools import partial
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    ref_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_prompt_length: int = field(
        default=512,
        metadata={
            "help": "Maximum target length."
        },
    )
    max_target_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum target length."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess_multi_turn(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
) -> Dict:
    conv = get_model_adapter(model_path).get_default_conv_template(model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conv.messages = []
    for j, sentence in enumerate(source['conversations']):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence["value"])
    input = conv.get_prompt()

    # Tokenize conversations

    # response_tokens = tokenizer(response, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    # response_labels = response_tokens.input_ids[0].clone()
    # response_masks = mask_labels(response, response_labels, tokenizer, conv)
    # response_tokens = response_tokens.input_ids[0][len(prompt_tokens["input_ids"][0])-1:]
    # response_masks = response_masks[len(prompt_tokens["input_ids"][0])-1:]
    
    tokenized_input = tokenizer(input, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    
    # if len(response_tokens) == 0:
        # response_tokens = torch.tensor([tokenizer.pad_token_id])
        # response_masks = torch.tensor([False])
        # print(source)

    if False:  # Inspect and check the correctness of masking
        z = chosen_labels.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(chosen)
        rank0_print(tokenizer.decode(z))
        z = rejected_labels.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(response)
        rank0_print(tokenizer.decode(z))
        exit()

    # return dict(
    #     prompt_input_ids=prompt_tokens['input_ids'][0].tolist(),
    #     response_input_ids=response_tokens.tolist(),
    #     response_mask=response_masks.tolist(),
    #     reward=source['reward'],
    # )
    return dict(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        reward=source["reward"],
    )



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    config.num_labels = 1

    # Load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    # if tokenizer.pad_token != tokenizer.unk_token and tokenizer.unk_token is not None:
    #     tokenizer.pad_token = tokenizer.unk_token
    # else:
    #     tokenizer.unk_token = tokenizer.pad_token
        
    # load data
    dataset = load_dataset("json", data_files=data_args.data_path)
    preprocess = partial(preprocess_multi_turn, tokenizer=tokenizer, model_path=model_args.model_name_or_path)
    train_dataset = dataset["train"].map(preprocess)
    
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        max_length=training_args.model_max_length,
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)
        
if __name__ == "__main__":
    train()

# if __name__ == "__main__":
#     # parser = HfArgumentParser((RewardConfig, ModelConfig))
#     # config, model_config = parser.parse_args_into_dataclasses()
#     # config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

#     ################
#     # Model & Tokenizer
#     ################
#     # torch_dtype = (
#     #     model_config.torch_dtype
#     #     if model_config.torch_dtype in ["auto", None]
#     #     else getattr(torch, model_config.torch_dtype)
#     # )
#     # quantization_config = get_quantization_config(model_config)
#     # model_kwargs = dict(
#     #     revision=model_config.model_revision,
#     #     trust_remote_code=model_config.trust_remote_code,
#     #     device_map=get_kbit_device_map() if quantization_config is not None else None,
#     #     quantization_config=quantization_config,
#     # )
#     # tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
#     # model = AutoModelForSequenceClassification.from_pretrained(
#     #     model_config.model_name_or_path, num_labels=1, **model_kwargs
#     # )
    

        

#     # if model_config.lora_task_type != "SEQ_CLS":
#     #     warnings.warn(
#     #         "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
#     #         " Make sure to pass --lora_task_type SEQ_CLS when using this script."
#     #     )

#     ################
#     # Dataset
#     ################
#     raw_datasets = load_dataset("Anthropic/hh-rlhf")
#     # Tokenize chosen/rejected pairs of inputs
#     # Adapt this section to your needs for custom datasets

#     def preprocess_function(examples):
#         new_examples = {
#             "input_ids_chosen": [],
#             "attention_mask_chosen": [],
#             "input_ids_rejected": [],
#             "attention_mask_rejected": [],
#         }
#         for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
#             tokenized_chosen = tokenizer(chosen)
#             tokenized_rejected = tokenizer(rejected)

#             new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
#             new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
#             new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
#             new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

#         return new_examples

#     # Preprocess the dataset and filter out examples that are longer than args.max_length
#     raw_datasets = raw_datasets.map(
#         preprocess_function,
#         batched=True,
#         num_proc=4,
#     )
#     raw_datasets = raw_datasets.filter(
#         lambda x: len(x["input_ids_chosen"]) <= config.max_length and len(x["input_ids_rejected"]) <= config.max_length
#     )
#     train_dataset = raw_datasets["train"]
#     eval_dataset = raw_datasets["test"]

#     ################
#     # Training
#     ################
#     trainer = RewardTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=config,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         peft_config=get_peft_config(model_config),
#     )
#     trainer.train()
#     trainer.save_model(config.output_dir)
#     trainer.push_to_hub()
#     metrics = trainer.evaluate()
#     trainer.log_metrics("eval", metrics)
#     print(metrics)
    
    
# from transformers import LlamaForSequenceClassification  