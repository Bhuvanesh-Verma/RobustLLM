import argparse
import os
import csv
import torch
import numpy as np
import random
import wandb
import yaml
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq, \
    BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, AutoModel, \
    AutoModelForSequenceClassification

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.utils.other import fsdp_auto_wrap_policy, prepare_model_for_kbit_training
from tqdm import tqdm
import torch.optim as optim
from prepare_data import prepare
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from torchtext.vocab import Vectors

import bitsandbytes as bnb
from torch import nn
from transformers import BertModel, BertTokenizer
from transformers.trainer_pt_utils import get_parameter_names
from accelerate.utils import ProjectConfiguration
from huggingface_hub import PyTorchModelHubMixin

import numpy as np


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


with open('configs/train_minmax.yaml') as file:
    config_data = yaml.safe_load(file)

# Replace 'path/to/glove.6B.300d.txt' with the actual path to your downloaded GloVe file
glove_path = '/ds/models/embedding_models/glove.6B.300d.txt'

# Load GloVe embeddings from the specified file
glove_vectors = Vectors(name=glove_path)


# Tokenizer
# tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def get_glove_embeddings(inputs, tokenizer):
    # Assuming 'input_ids' in inputs is the tokenized version of the sentences
    # Convert tokenized sentences to list of words
    # sentences = [tokenizer.decode(ids) for ids in inputs['input_ids'].tolist()]

    # Get GloVe embeddings for each word in the sentences
    # embeddings = [glove_vectors[word] for sentence in sentences for word in sentence.split()]

    # embeddings = [glove_vectors[word] for ids in inputs['input_ids'].tolist() for word in tokenizer.decode(ids)]
    embeddings = []
    for ids in inputs['input_ids'].tolist():
        res = []
        for word in tokenizer.decode(ids).split():
            if word == '[PAD]':
                continue
            res.append(glove_vectors[word])
        res = torch.stack(res).mean(dim=0)
        embeddings.append(res),

    # Reshape to match the batch size and sequence length
    embeddings = torch.stack(embeddings).view(inputs['input_ids'].shape[0], 300).to('cuda')
    # Create one-hot encoded labels
    # labels_onehot = torch.eye(2).to('cuda')[inputs['label'].squeeze(0)]  # Assumes 'label' is a tensor of shape (batch_size,)
    labels_oh = torch.zeros(embeddings.shape[0], 2).to('cuda').scatter_(1, inputs['label'].unsqueeze(1), 1)

    embeddings = torch.cat((embeddings, labels_oh), dim=1)
    return embeddings


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


class AuxiliaryModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size_1, hidden_size_2, hidden_size_3, output_size=1, input_size=300, num_labels=2, ):
        super(AuxiliaryModel, self).__init__()
        self.layer1 = nn.Linear(input_size + num_labels, hidden_size_1)
        next_size = hidden_size_1
        self.layer2 = None
        if hidden_size_2 != 0:
            self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
            next_size = hidden_size_2
            self.layer3 = None
            if hidden_size_3 != 0:
                self.layer3 = nn.Linear(hidden_size_2, hidden_size_3)
                next_size = hidden_size_3
        self.activation = nn.Tanh()
        self.layer4 = nn.Linear(next_size, output_size)  # Concatenate labels to the input

    def forward(self, x):
        # Concatenate labels to the input
        # x = torch.cat([x, labels.reshape(1,-1)], dim=1)

        x = self.layer1(x)
        x = self.activation(x)
        if self.layer2 is not None:
            x = self.layer2(x)
            x = self.activation(x)
            if self.layer3 is not None:
                x = self.layer3(x)
                x = self.activation(x)
        x = self.layer4(x)
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


def get_optim(model, config_data, lr):
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
    optimizer_kwargs["lr"] = lr
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(config_data['training']['optim']['beta1'], config_data['training']['optim']['beta2']),
        eps=config_data['training']['optim']['epsilon'],
        lr=lr,
    )

    return adam_bnb_optim


def remove_large_samples(tokenizer, dataset, max_length):
    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer})
    exclude_idx = []
    old_len = len(dataset)

    for i, data in enumerate(tokenized_dataset):
        if len(data['input_ids']) > max_length:
            exclude_idx.append(i)

    # create new dataset excluding those idx
    new_dataset = dataset.select(
        (
            i for i in range(len(dataset))
            if i not in set(exclude_idx)
        )
    )
    new_len = len(new_dataset)

    print(f'{old_len - new_len} instance removed')


def train():
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
        # config=config_data,
    )
    print(f'Base Config: {config_data}')
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

    train_dataset = prepare(split=f"{config_data['data']['train_name']}.json",
                            prompt_type=config_data['data']['prompt_type'])
    # train_dataset = prepare_bionli(prompt_type=config_data['data']['prompt_type'])
    val_dataset = prepare(split=f"{config_data['data']['val_name']}.json",
                          prompt_type='eval')

    e = [i for i, l in enumerate(train_dataset['label_cls']) if l == 0]
    c = [i for i, l in enumerate(train_dataset['label_cls']) if l == 1]
    a = random.sample(e, 200) + random.sample(c, 200)
    random.shuffle(a)

    train_dataset = train_dataset.select(a)

    base_model = config_data['base_model']
    label2id = {label: i for i, label in enumerate(list(set(train_dataset['label'])))}
    id2label = {i: label for label, i in label2id.items()}

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
    # model.config.pad_token_id = model.config.eos_token_id
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
    num_epochs = int(wandb.config['epoch'])

    with accelerator.main_process_first():
        train_tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["label", 'text', 'id_'],
            fn_kwargs={'max_length': int(config_data['tokenizer']['max_length']),
                       'tokenizer': tokenizer, "padding": config_data['tokenizer']['padding'],
                       "truncation": config_data['tokenizer']['truncation']}
        )

        val_tokenized_datasets = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["label", 'text', 'id_'],
            fn_kwargs={'max_length': int(config_data['tokenizer']['max_length']),
                       'tokenizer': tokenizer, "padding": config_data['tokenizer']['padding'],
                       "truncation": config_data['tokenizer']['truncation']}
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    y = torch.tensor(train_tokenized_datasets['label_cls'])
    sampler = StratifiedSampler(class_vector=y, batch_size=batch_size)
    train_dataloader = DataLoader(
        train_tokenized_datasets, sampler=sampler, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        val_tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )

    optimizer = get_optim(model, config_data, lr=wandb.config['learner_lr'])

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # auxiliary_model = Auxiliary(hidden_dim=wandb.config['aux_hidden_size'])
    auxiliary_model = AuxiliaryModel(
        hidden_size_1=wandb.config['aux_hidden_size1'],
        hidden_size_2=wandb.config['aux_hidden_size2'],
        hidden_size_3=wandb.config['aux_hidden_size3'])

    auxiliary_optimizer = optim.Adam(auxiliary_model.parameters(), lr=wandb.config['aux_lr'])

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    learner, auxiliary_model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, auxiliary_optimizer = accelerator.prepare(
        model, auxiliary_model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, auxiliary_optimizer
    )

    callback = EarlyStoppingCallback(patience=wandb.config['patience'])
    overall_step = 0
    best_vloss = 1000
    best_f1 = 0
    best_run = {'epoch': 0, 'step': 0, 'f1': 0}
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        # If so, we break the loop
        if accelerator.check_trigger():
            print('Ran out of patience')
            break
        learner.train()
        running_loss = 0
        last_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # If so, we break the loop
            if accelerator.check_trigger():
                print('Ran out of patience')
                break
            # input_ids_batch, attention_mask_batch, labels_batch = batch
            input_ids_batch = batch['input_ids']
            attention_mask_batch = batch['attention_mask']
            labels_batch = batch['label_cls']

            inputs_example = {'input_ids': input_ids_batch, 'label': labels_batch}

            glove_embeddings = get_glove_embeddings(inputs_example, tokenizer)
            weights = auxiliary_model(glove_embeddings)
            weights = F.sigmoid(weights)
            weights = weights / weights.mean()
            weights = weights + 1

            outputs = learner(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=batch['labels'])
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

        last_loss = running_loss / step

        learner.eval()
        runnning_vloss = 0
        preds = []
        golds = []
        for e_step, e_batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                input_ids_batch = e_batch['input_ids']
                attention_mask_batch = e_batch['attention_mask']
                labels_batch = e_batch['label_cls']
                outputs = learner(input_ids=input_ids_batch, attention_mask=attention_mask_batch,
                                  labels=e_batch['labels'])
                probs = torch.softmax(outputs['logits'].detach().cpu(), dim=2)
                toks = torch.argmax(probs, dim=2)
                golds.extend([id2label[int(g)] for g in e_batch['label_cls']])
                for o, t in enumerate(toks):
                    s = tokenizer.decode(t)
                    s = s.split('### Response:')
                    if len(s) != 0:
                        s = s[-1]
                        if 'yes' in s.lower():
                            preds.append(id2label[0])
                        elif 'no' in s.lower():
                            preds.append(id2label[1])
                        else:
                            if int(e_batch['label_cls'][o]) == 0:
                                preds.append(id2label[1])
                            else:
                                preds.append(id2label[0])
                    else:
                        if int(e_batch['label_cls'][o]) == 0:
                            preds.append(id2label[1])
                        else:
                            preds.append(id2label[0])

            e_loss = outputs.loss
            runnning_vloss += e_loss.detach().float()
        avg_eval_loss = runnning_vloss / (e_step + 1)

        scores = compute_metrics_decoded(golds, preds)
        f1_score = scores['macro_f1']
        print(classification_report(golds, preds))
        if f1_score > best_f1:
            best_f1 = f1_score
            best_run = {'epoch': epoch, 'step': step, 'f1': best_f1}

        # print(f'Step: {step+1} Avg Train Loss: {last_loss} Avg Eval Loss: {avg_eval_loss} Val F1 Macro: {f1_score}')
        learner.train()
        wandb.log(
            {
                # "epoch": epoch,
                "step": overall_step,
                # "train_loss": learner_loss,
                "train_avg_loss": last_loss,
                # "val_loss": e_loss,
                "val_avg_loss": avg_eval_loss,
                "macro_f1": f1_score,
            }
        )
        # Check if we should stop the training on any processes
        if callback.check_early_stopping(avg_eval_loss):
            accelerator.set_trigger()

        # If so, we break the loop
        if accelerator.check_trigger():
            print('Ran out of patience')
            break

        print(f'Epoch: {epoch} Training Loss: {last_loss} Evaluation Loss: {avg_eval_loss} Evaluation F1: {f1_score}')
    print(f'Best Configuration: \n {best_run}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training models')

    parser.add_argument(
        '-sweep_config',
        help='Path to train config file',
        type=str, default='configs/hpt.yaml',
    )
    parser.add_argument(
        '-count',
        help='Number of sweep run',
        type=int, default=2,
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.sweep_config) as file:
        sweep_config = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_config, project=config_data['logger']['project'],
                           entity=config_data['logger']['entity'])
    wandb.agent(sweep_id, function=train, count=args.count)
