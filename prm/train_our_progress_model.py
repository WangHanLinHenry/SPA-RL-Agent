# This is an example of data processing for webshop environment
# You can change the data path according to different environments (e.g., alfworld, virtualhome)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer
)
import transformers
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Optional, Sequence

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

max_length = 4096

# Load pre-trained model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("ckt/llama3b_webshop_sft", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("ckt/llama3b_webshop_sft", trust_remote_code=True, torch_dtype=torch.bfloat16)

class prm_model(PreTrainedModel):
    def __init__(self, base_model, vocab_size=32000):
        # Use base_model's configuration
        super().__init__(base_model.config)
        # Use base_model as a submodule
        self.backbone = base_model
        self.LN = nn.Linear(vocab_size, 1).to(torch.bfloat16)
        
        # Ensure correct configuration
        self.config.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, gpt_unmask=None, labels=None):
        # Get logits for all tokens
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Initialize list to store final predictions for each sample
        batch_predictions = []
        
        # Iterate through each sample
        for i in range(outputs.size(0)):
            # Get labels for current sample
            sample_labels = gpt_unmask[i]
            # Find all non-100 positions
            valid_indices = torch.where(sample_labels != -100)[0]
            
            if len(valid_indices) == 0:
                # If no valid tokens, add zero value
                batch_predictions.append(torch.zeros(1, device=outputs.device, dtype=outputs.dtype))
                continue
                
            # Find indices of last token in each turn
            turn_end_indices = []
            
            # Iterate through all non-100 positions
            for j in range(1, len(valid_indices)):
                # If current token and previous token are not consecutive, previous token was last in turn
                if valid_indices[j] - valid_indices[j-1] > 1:
                    turn_end_indices.append(valid_indices[j-1])
            
            # Add last token of final turn
            turn_end_indices.append(valid_indices[-1])
            
            # Don't take first index since it's ok
            # if len(turn_end_indices) > 1:
            #     turn_end_indices = turn_end_indices[1:]
            
            # Get logits for last token of each turn
            turn_logits = []
            for idx in turn_end_indices:
                turn_logits.append(outputs[i, idx, :])
            
            # Stack logits from all turns
            turn_logits = torch.stack(turn_logits)
            # Pass through linear layer
            turn_values = self.LN(turn_logits)
            # Sum values from all turns
            sample_prediction = turn_values.sum()
            batch_predictions.append(sample_prediction.unsqueeze(0))  # Ensure each prediction is tensor of shape [1]
        
        # Stack predictions from all samples
        value_outputs = torch.cat(batch_predictions)  # Use cat instead of stack
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(value_outputs, labels)
            
        return {
            'loss': loss,
            'predictions': value_outputs
        }

def read_json(source):
    print(f"Reading file: {source}")
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        # Get total number of lines
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer
        for line in tqdm(f, total=total_lines, desc="Reading progress"):
            json_list.append(json.loads(line))
    return json_list


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
) -> Dict:
    conv = get_model_adapter(model_path).get_default_conv_template(model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # print("pad token:" + tokenizer.pad_token)

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    
    if 'Llama-3.2-3B-Instruct' in model_path or 'Llama-3.1-8B-Instruct' in model_path:

        sep2 = "<|eot_id|>"
        sep = "<|end_header_id|>"
        
        targets = targets[:,1:]
        input_ids = input_ids[:,1:]
        
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            
            turns = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if i%2 == 0:
                    if i == 0:
                        instruction_len = len(tokenizer(turn).input_ids[1:])
                        target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                        cur_len += instruction_len
                    else:
                        instruction_len = len(tokenizer(turn).input_ids[1:])
                        target[cur_len: cur_len + instruction_len+1] = IGNORE_TOKEN_ID
                        cur_len += instruction_len+1
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids[1:])
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids[1:])
                    target[cur_len: cur_len +2] = IGNORE_TOKEN_ID
                    cur_len += turn_len+1
            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(conversation)
                rank0_print(tokenizer.decode(z))
                exit()

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            gpt_unmask=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    
    if conv.sep_style == SeparatorStyle.LLAMA3:

        sep2 = "<|eot_id|>"
        sep = "<|end_header_id|>"
        
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            
            turns = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if i%2 == 0:
                    if i == 0:
                        instruction_len = len(tokenizer(turn).input_ids)
                        target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                        cur_len += instruction_len
                    else:
                        instruction_len = len(tokenizer(turn).input_ids)
                        target[cur_len: cur_len + instruction_len+1] = IGNORE_TOKEN_ID
                        cur_len += instruction_len+1
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids)
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                    target[cur_len: cur_len + +2] = IGNORE_TOKEN_ID
                    cur_len += turn_len+1
            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(conversation)
                rank0_print(tokenizer.decode(z))
                exit()

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    else:
        raise NotImplementedError

    # Mask targets. Only compute loss on the assistant outputs.
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break

            # remove <s>
            turn_len = len(tokenizer(turn).input_ids) - 1

            parts = turn.split(sep)

            if len(parts) != 2:
                break
            parts[0] += sep
            
            # remove <s> and the "_" in the end
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # magic number for vicuna, since different subtoken for "USER"
            if i != 0 and conv.roles[0] == 'USER':
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            # add the length of turn sep
            if conv.sep2 == '</s>':
                cur_len += turn_len + 1 
            elif conv.sep2 == ' </s><s>':
                cur_len += turn_len + 3
            else:
                raise NotImplementedError
            
            # magic number for vicuna, since different subtoken for "USER"
            if i != 0 and conv.roles[0] == 'USER':
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(conversation)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path)
        
        self.input_ids = data_dict["input_ids"]
        self.gpt_unmask = data_dict["gpt_unmask"]
        self.attention_mask = data_dict["attention_mask"]
        
        self.labels = []
        for each_piece in raw_data:
            self.labels.append(each_piece['agent_final_reward'])
        self.labels = torch.tensor(self.labels, dtype=torch.bfloat16)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            gpt_unmask=self.gpt_unmask[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i]
        )

train_data = json.load(open("exploration/webshop/exploration.json", "r"))
val_data = json.load(open("exploration/webshop/exploration_tiny.json", "r"))
test_data = json.load(open("exploration/webshop/exploration_tiny.json", "r"))

# Small batch data trial
# train_data = train_data[:100]

train_dataset = SupervisedDataset(train_data, tokenizer, model_path="Llama-3.2-3B-Instruct")
val_dataset = SupervisedDataset(val_data, tokenizer, model_path="Llama-3.2-3B-Instruct")
test_dataset = SupervisedDataset(test_data, tokenizer, model_path="Llama-3.2-3B-Instruct")

# Define function to compute metrics
def compute_metrics(eval_pred):
    # eval_pred is a tuple containing (predictions, labels)
    predictions, labels = eval_pred
    
    # Calculate MSE and MAE
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    # Calculate accuracy (proportion of predictions within 0.1 of true value)
    accuracy = np.mean(np.abs(predictions - labels) <= 0.1)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "accuracy": float(accuracy)
    }
    

# Create model instance
print("Creating model instance...")
vocab_size = base_model.config.vocab_size
model = prm_model(base_model, vocab_size)

# DeepSpeed configuration
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": "auto",  # Set to auto to match TrainingArguments
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-6,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-6,
            "warmup_num_steps": 500
        }
    },
    "gradient_clipping": 1.0,
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "offload_optimizer": {  # Enable optimizer CPU offload
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {  # Enable parameter CPU offload
            "device": "cpu",
            "pin_memory": True
        }
    },
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
    "fp16": {
        "enabled": False
    },
    "amp": {
        "enabled": False
    }
}

# Define training parameters
print("Configuring training parameters...")
training_args = TrainingArguments(
    output_dir="./records/progress_model_virtualhome",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="no",
    # eval_steps=1,
    save_strategy="no",  # Save by epoch
    # save_total_limit=1,  # Only save last checkpoint
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=3e-6,
    bf16=True,
    save_safetensors=False,  # Disable safetensors saving
    deepspeed=deepspeed_config,
    gradient_accumulation_steps=2  # Set gradient accumulation steps
)

# Create Trainer
print("Creating Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()

# Evaluate model
# print("Starting model evaluation...")
# test_results = trainer.evaluate(test_dataset)
# print("Test results:", test_results)

# Save model
print("Saving model...")
# Save complete model state
model_save_path = "ckt/llama3b_webshop_prm"
os.makedirs(model_save_path, exist_ok=True)

# Save base model
base_model_save_path = os.path.join(model_save_path, "our_base_model")
base_model.save_pretrained(base_model_save_path)
tokenizer.save_pretrained(base_model_save_path)

# Save complete model state
model_state_dict = {
    'model_state_dict': model.state_dict(),
    'config': model.config
}
torch.save(model_state_dict, os.path.join(model_save_path, "our_model_state.pt"))
print(f"Model saved to: {model_save_path}")
