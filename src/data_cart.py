import argparse
import os
import csv
import torch
import wandb
import yaml
import random
import json
from accelerate import Accelerator
from collections import defaultdict as dd
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq, \
    BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy, prepare_model_for_kbit_training
from tqdm import tqdm

from prepare_data import prepare, prepare_bionli, prepare_med_nli
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from itertools import cycle
from peft import PeftModel
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
from accelerate.utils import ProjectConfiguration
from sklearn.model_selection import StratifiedKFold


def tokenize_function(example, tokenizer=None, max_length=None, truncation=None, padding='do_not_pad'):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(example["text"], max_length=max_length, truncation=truncation, padding=padding)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def compute_metrics_decoded(decoded_labs, decoded_preds):
    metrics = {
        "micro_f1": f1_score(decoded_labs, decoded_preds, average="micro"),
        "macro_f1": f1_score(decoded_labs, decoded_preds, average="macro"),
        "precision": precision_score(decoded_labs, decoded_preds, average="micro"),
        "recall": recall_score(decoded_labs, decoded_preds, average="micro"),
        "accuracy": accuracy_score(decoded_labs, decoded_preds),
    }

    return metrics


def get_optim(model, config_data):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": config_data['training']['optim']['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (config_data['training']['optim']['beta1'], config_data['training']['optim']['beta2']),
        "eps": config_data['training']['optim']['epsilon'],
    }
    optimizer_kwargs["lr"] = config_data['training']['optim']['learning_rate']
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(config_data['training']['optim']['beta1'], config_data['training']['optim']['beta2']),
        eps=config_data['training']['optim']['epsilon'],
        lr=config_data['training']['optim']['learning_rate'],
    )

    return adam_bnb_optim


def remove_large_samples(tokenizer, dataset, max_length):
    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer})
    exclude_idx = []
    old_len = len(dataset)

    for i, data in enumerate(tokenized_dataset):
        if len(data['input_ids']) > max_length:
            exclude_idx.append(i)

    # create new dataset exluding those idx
    new_dataset = dataset.select(
        (
            i for i in range(len(dataset))
            if i not in set(exclude_idx)
        )
    )
    new_len = len(new_dataset)

    print(f'{old_len - new_len} instance removed')


def train(config_data):
    log_dir = os.path.join(config_data['logger']['dir'], config_data['training']['run_name'])
    output_dir = os.path.join(config_data['training']['output_dir'], config_data['training']['run_name'])
    project_config = ProjectConfiguration(
        total_limit=5,
        project_dir=output_dir,
        automatic_checkpoint_naming=False,  # iteration=1,
        logging_dir=log_dir
    )
    accelerator = Accelerator(mixed_precision="bf16", project_config=project_config)

    train_dataset = prepare(split='train.json')

    base_model = config_data['base_model']

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        use_flash_attention_2=False,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    # model.config.pretraining_tp = 1

    if 'load' in config_data:
        print('Loading Finetuned Checkpoint ...')
        model = PeftModel.from_pretrained(model, config_data['load']['ckpt'], is_trainable=True)
    else:
        lora_config = config_data["lora_config"]
        config = LoraConfig(
            r=int(lora_config['r']),
            lora_alpha=int(lora_config['lora_alpha']),
            target_modules=lora_config['target_modules'],
            bias="none",
            lora_dropout=float(lora_config['lora_dropout']),  # Conventional
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    if config_data['data']['reduce']:
        train_dataset = remove_large_samples(tokenizer, train_dataset,
                                             config_data['tokenizer']['max_length'])

    batch_size = config_data['training']['batch_size']
    gradient_accumulation_steps = 1
    num_epochs = config_data['training']['num_train_epochs']
    # accelerator.print(model.print_trainable_parameters())

    with accelerator.main_process_first():
        train_tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["label", 'text'],
            fn_kwargs={'max_length': int(config_data['tokenizer']['max_length']),
                       'tokenizer': tokenizer, "padding": config_data['tokenizer']['padding'],
                       "truncation": config_data['tokenizer']['truncation']}
        )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_tokenized_datasets,
        collate_fn=data_collator,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        # batch_sampler=StratifiedBatchSampler(torch.tensor(train_tokenized_datasets['label_cls']), batch_size=batch_size),
    )

    optimizer = get_optim(model, config_data)
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr = config_data['training']['optim']['learning_rate'])

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )


    data_cart = dd(lambda: dd(lambda: dd(list)))
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        print(f'Epoch: {epoch}')
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            label_vocab_key = batch['input_ids'][:, -4]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            probs = torch.softmax(outputs['logits'].detach().cpu(), dim=2)
            toks = torch.argmax(probs, dim=2)

            for k, tok in enumerate(toks):
                if tok[-5] in [5613, 2501]:
                    prob = float(probs[k, -5, label_vocab_key[k]])
                    idx = int(batch['id'][k].detach().cpu())
                    data_cart['nlict'][epoch][idx].append(prob)
                    wandb.log({'nlict_idx': idx, f'nlict_prob_{epoch}': prob})

            loss = outputs.loss
            running_loss += loss.detach().float()

            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad()
                # model.zero_grad()

    json.dump(data_cart, open('data_cart_gen.json', 'w+'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='configs/train.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    wandb.init(
        # Set the project where this run will be logged
        project=config_data['logger']['project'],
        # Track hyperparameters and run metadata
        entity=config_data['logger']['entity'],
        dir=config_data['logger']['dir'],
        mode=config_data['logger']['mode'],
        config=config_data,
    )

    train(config_data)
