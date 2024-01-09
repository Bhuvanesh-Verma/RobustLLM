import argparse
import os
import random

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from huggingface_hub import PyTorchModelHubMixin
from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils.other import fsdp_auto_wrap_policy, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, AutoModel
from transformers.trainer_pt_utils import get_parameter_names

from prepare_data import prepare


class EarlyStoppingCallback:
    "A callback class that helps with early stopping"

    def __init__(self, min_delta=0, patience=5):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float("inf")

    def check_early_stopping(self, eval_loss):
        delta = self.lowest_loss - eval_loss
        if delta >= self.min_delta:
            self.lowest_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Auxiliary(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_dim=256):
        super(Auxiliary, self).__init__()
        # self.embedding_layer = nn.Embedding(300, 768)  # Assuming GloVe embeddings size 300
        self.model = AutoModel.from_pretrained("models/SGPT-125M-weightedmean-nli-bitfit")
        self.mlp = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, y):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                             return_dict=True)
        last_hidden_state = outputs['last_hidden_state']
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask
        # pooled_output = outputs['pooler_output']
        x = self.mlp(embeddings)
        # combined_input = torch.cat((x, y), dim=1)
        return x


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
        automatic_checkpoint_naming=False,
        # iteration=1,
        logging_dir=log_dir
    )
    accelerator = Accelerator(mixed_precision="bf16", project_config=project_config)

    train_dataset = prepare(split='train.json', prompt_type=config_data['data']['prompt_type'],
                            mode=config_data['data']['mode'])
    val_dataset = prepare(split='dev.json', prompt_type=config_data['data']['prompt_type'],
                          mode=config_data['data']['mode'])

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

        val_dataset = remove_large_samples(tokenizer, val_dataset,
                                           config_data['tokenizer']['max_length'])

    batch_size = config_data['training']['batch_size']
    gradient_accumulation_steps = 1
    num_epochs = config_data['training']['num_train_epochs']

    with accelerator.main_process_first():
        train_tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["label", 'text'],
            fn_kwargs={'max_length': int(config_data['tokenizer']['max_length']),
                       'tokenizer': tokenizer, "padding": config_data['tokenizer']['padding'],
                       "truncation": config_data['tokenizer']['truncation']}
        )

        val_tokenized_datasets = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["label", 'text'],
            fn_kwargs={'max_length': int(config_data['tokenizer']['max_length']),
                       'tokenizer': tokenizer, "padding": config_data['tokenizer']['padding'],
                       "truncation": config_data['tokenizer']['truncation']}
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_tokenized_datasets, shuffle=True, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        val_tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )

    optimizer = get_optim(model, config_data)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    auxiliary_model = Auxiliary(hidden_dim=config_data['aux']['hidden_size'])
    auxiliary_optimizer = optim.Adam(auxiliary_model.parameters(), lr=config_data['aux']['lr'])

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    learner, auxiliary_model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, auxiliary_optimizer = accelerator.prepare(
        model, auxiliary_model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, auxiliary_optimizer
    )
    callback = EarlyStoppingCallback(patience=config_data['patience'])
    overall_step = 0
    best_vloss = 1000
    best_run = {'epoch': 0, 'step': 0, 'vloss': 1000}
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        learner.train()
        running_loss = 0
        last_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids_batch, attention_mask_batch, labels_batch = batch

            weights = auxiliary_model(batch['input_ids'], batch['attention_mask'], batch['labels'])
            weights = F.sigmoid(weights)
            weights = weights / weights.mean()
            weights = weights + 1

            outputs = learner(**batch)
            loss = outputs.loss
            learner_loss = torch.mean(weights.detach() * loss)

            # accelerator.backward(learner_loss)

            auxiliary_loss = -torch.mean(weights * loss.detach())

            # accelerator.backward(auxiliary_loss)
            running_loss += learner_loss.detach().float()

            if step % gradient_accumulation_steps == 0:
                learner_loss.backward()
                auxiliary_loss.backward()

                optimizer.step()
                lr_scheduler.step()
                auxiliary_optimizer.step()

                optimizer.zero_grad()
                auxiliary_optimizer.zero_grad()

            if (step + 1) % 25 == 0:
                overall_step = +25
                last_loss = running_loss / 25
                running_loss = 0
                # accelerator.save_state(output_dir=f'{output_dir}/epoch-{epoch}_step-{step+1}')
                slearner = accelerator.unwrap_model(learner)
                slearner.save_pretrained(f'{output_dir}/learner/epoch-{epoch}_step-{step + 1}')
                saux = accelerator.unwrap_model(auxiliary_model)
                saux.save_pretrained(f'{output_dir}/aux/epoch-{epoch}_step-{step + 1}')
                learner.eval()
                runnning_vloss = 0
                for e_step, e_batch in enumerate(tqdm(eval_dataloader)):
                    with torch.no_grad():
                        outputs = learner(**e_batch)
                    e_loss = outputs.loss
                    runnning_vloss += e_loss.detach().float()
                avg_eval_loss = runnning_vloss / (e_step + 1)
                if avg_eval_loss < best_vloss:
                    best_vloss = avg_eval_loss
                    best_run = {'epoch': epoch, 'step': step, 'vloss': best_vloss}

                print(f'Step: {step + 1} Avg Train Loss: {last_loss} Avg Eval Loss: {avg_eval_loss}')
                learner.train()
                wandb.log(
                    {
                        # "epoch": epoch,
                        "step": overall_step,
                        # "train_loss": learner_loss,
                        "train_avg_loss": last_loss,
                        # "val_loss": e_loss,
                        "val_avg_loss": avg_eval_loss,
                    }
                )
                # Check if we should stop the training on any processes
                if callback.check_early_stopping(avg_eval_loss):
                    accelerator.set_trigger()

                # If so, we break the loop
                if accelerator.check_trigger():
                    print('Ran out of patience')
                    break
        print(f'Epoch: {epoch} Training Loss: {last_loss} Evaluation Loss: {avg_eval_loss}')

        wandb.log(
            {
                "epoch": epoch,
                # "step": step,
                # "total_train_loss": loss,
                "train_loss": last_loss,
                # "val_loss": e_loss,
                "val_loss": avg_eval_loss,
            }
        )
    print(f'Best Configuration: \n {best_run}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='configs/train_1.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    torch.manual_seed(config_data['seed'])
    random.seed(config_data['seed'])
    np.random.seed(config_data['seed'])

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
