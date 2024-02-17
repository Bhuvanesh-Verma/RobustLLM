import os
import json
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
import numpy as np

from accelerate.utils import ProjectConfiguration
import torch
from collections import defaultdict as dd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from pathlib import Path

from prepare_data import prepare
from peft import PeftModel

from evaluation import evaluate

EVAL_PROMPT = """
<s>### Instruction:
Read the input text and answer the following question with Yes or No.

### Input:
{premise}

Question: Does this imply that {hypothesis}?

### Response:
"""


def get_input_text_p(premise, hypothesis, PROMPT):
    return PROMPT.format(premise=premise, hypothesis=hypothesis)


def get_predicted_label(instance_data):
    # Get the 'prediction' sub-dictionary
    prediction_data = instance_data.get('prediction', {})

    # Initialize variables to store the predicted label and its count
    predicted_label = None
    max_score = 0

    # Iterate through the prediction data to find the label with the highest count
    for label, label_data in prediction_data.items():
        count = label_data.get('count', 0)
        scores = 1  # sum(label_data.get('score', 0))
        score = count * scores
        if score > max_score:
            max_score = score
            predicted_label = label

    return predicted_label


def tokenize_function(example, tokenizer=None, max_length=None, truncation=None, padding='do_not_pad'):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(example["text"], max_length=max_length, truncation=truncation, padding=padding)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def get_data():
    split = "final_test.json"
    data = json.load(open(f"Complete_dataset/{split}"))
    files = os.listdir("Complete_dataset/CT json/")
    files.remove(".DS_Store")

    files_data = {file[:-5]: json.load(open(f"Complete_dataset/CT json/{file}")) for file in files}

    data_expanded = []
    for _id, value in data.items():
        temp = {}
        temp["id"] = _id
        p_nctid = value["Primary_id"]
        s_nctid = value.get("Secondary_id")
        section_id = value["Section_id"]
        statement = value["Statement"]
        primary_evidence = files_data[p_nctid][section_id]
        temp["statement"] = statement
        temp["primary_evidence"] = primary_evidence

        if s_nctid is not None:
            secondary_evidence = files_data[s_nctid][section_id]
            temp["secondary_evidence"] = secondary_evidence

        data_expanded.append(temp)

    samples = []
    for sample in data_expanded:
        primary_evidence = "".join(sample['primary_evidence'])
        sentence = f"Primary trial evidence are {primary_evidence}"
        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary_evidence = "".join(sample['secondary_evidence'])
            sentence = f"{sentence} Secondary trial evidence are {secondary_evidence}"
        input_text = get_input_text_p(sentence, sample['statement'], EVAL_PROMPT)
        if 'label' in sample:
            temp = {"text": input_text, "label": sample['label'], "id": sample['id']}
        else:
            temp = {"text": input_text, "id": sample['id']}
        samples.append(temp)

    return samples


with open('configs/inference.yaml') as file:
    config_data = yaml.safe_load(file)

log_dir = os.path.join(config_data['logger']['dir'], config_data['training']['run_name'])
output_dir = os.path.join(config_data['training']['output_dir'], config_data['training']['run_name'])
project_config = ProjectConfiguration(
    total_limit=5,
    project_dir=output_dir,
    automatic_checkpoint_naming=False,  # iteration=1,
    logging_dir=log_dir
)
accelerator = Accelerator(mixed_precision="bf16", project_config=project_config)

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_name = "models/Mistral-7B-Instruct-v0.2"
# model_name = "models/Mistral-7B-v0.1"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto",
)
# model.config.pretraining_tp = 1
model.config.pad_token_id = model.config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = get_data()
label2id = {'Entailment': 0, 'Contradiction': 1}
id2label = {i: label for label, i in label2id.items()}

model2path = {
    "nlict-ft": 'results/acc_mistral_2048_truncate/best_epoch',
    "nlict-ft-dataset1": 'results/acc_mistral_2048_truncate_dataset_1/best_epoch',
    "mednli-ft": 'results/acc_mistral_512_truncate_mednli_cls/best_epoch',
    "mednli-ft-nlict": 'results/acc_mistral_4096_truncate_mednli_ct_ft/best_epoch',
    "mednli-nlict-ft-acr": 'results/acc_mistral_4096_truncate_mednli_ct_ft_acr_pert/best_epoch',
    "mednli-nlict-ft-num": 'results/acc_mistral_4096_truncate_mednli_ct_ft_num_pert/best_epoch1',
    "mednli-nlict-ft-acr-num": 'results/acc_mistral_4096_truncate_mednli_ct_ft_acr_num_pert/best_epoch1',
    "mednli-ft-nlict-acr-num": 'results/acc_mistral_4096_truncate_mednli_ft_ct_acr_num_pert/best_epoch',
    "minmax-mednli-ft-nlict": 'results/acc_minmax_ft_mistral_4096_truncate_mednli_ft/learner/best_epoch',
    "minmax-mednli-ft-nlict-bc": 'results/acc_minmax_ft_mistral_4096_truncate_mednli_ft_best_config/learner/best_epoch1',
    "minmax-mednli-nlict-ft-acr-num-bc": 'results/acc_minmax_ft_mistral_4096_truncate_mednli_ft_acr_num_pert/learner/epoch-0_step-150'
}

for model_type, model_path in tqdm(model2path.items()):
    ft_model = PeftModel.from_pretrained(model, model_path)

    results = dd(lambda: dd(lambda: dd(lambda: {'count': 0, 'score': []})))
    for epoch in range(3):
        res = dd()
        labels = []
        for sample in dataset:
            id_ = sample['id']
            inputs = tokenizer(sample['text'], return_tensors="pt").to('cuda')
            with torch.inference_mode():
                outputs = ft_model.generate(
                    input_ids=inputs["input_ids"].to("cuda"),
                    attention_mask=inputs['attention_mask'].to('cuda'),
                    max_new_tokens=2,  # len(inputs["input_ids"])+8,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.1,
                    output_scores=True, return_dict_in_generate=True,
                )
                transition_scores = ft_model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                input_length = 1 if ft_model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]
                for tok, score in zip(generated_tokens[0], transition_scores[0]):
                    if tokenizer.decode(tok).lower() in ['no', 'yes', 'entailment', 'contradaction']:
                        res[id_] = {"score": float(np.exp(score.cpu())), "output": tokenizer.decode(tok)}

        for id_, r in res.items():
            word = r['output']
            if 'yes' in word.lower():
                results[id_]['prediction']['Entailment']['count'] += 1
                results[id_]['prediction']['Entailment']['score'].append(r['score'])
            elif 'no' in word.lower():
                results[id_]['prediction']['Contradiction']['count'] += 1
                results[id_]['prediction']['Contradiction']['score'].append(r['score'])

    final_res = dd()
    for instance_id, instance_data in results.items():
        predicted_label = get_predicted_label(instance_data)
        final_res[instance_id] = {"Prediction": predicted_label}

    output_dir = Path(f'results/{model_type}')
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(final_res, open(os.path.join(output_dir, 'results.json'), 'w+'), indent=4)
    # evaluate(final_res, output_dir=output_dir)

