import json
import os
from datasets import Dataset

ZERO_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 2 classes. The list of classes is provided below, where the classes are separated by commas: 

[Entailment, Contradiction]

From the above list of classes, select only one class that the provided sentence can be classified into. The sentence will be delimited with triple backticks. Once again, only predict the class from the given list of classes. Do not predict anything else.

### Sentence: ```{sentence}```
### Class:
"""

NEW_PROMPT = """{premise}'
Question: Does this imply that '{hypothesis}' ? Answer Yes or No.
### Answer:
"""

PROMPT_INST = """<s>[INST]{premise}
###Question: Does this imply that {hypothesis} ? Answer Yes or No.[/INST]
###Answer:
"""

TRAIN_PROMPT = """
<s>### Instruction:
Read the input text and answer the following question with Yes or No.

### Input:
{premise}

Question: Does this imply that {hypothesis}?

### Response:
{label}</s>
"""

EVAL_PROMPT = """
<s>### Instruction:
Read the input text and answer the following question with Yes or No.

### Input:
{premise}

Question: Does this imply that {hypothesis}?

### Response:
"""

TRAIN_PROMPT_BASE = """{premise}
Question: Does this imply that '{hypothesis}' ? Answer Yes or No.
### Answer: {label}
"""

CLASS_PROMPT = """
<s>Premise: {premise}

Hypothesis: {hypothesis}</s>
"""


def get_input_text_p(premise, hypothesis, PROMPT):
    return PROMPT.format(premise=premise, hypothesis=hypothesis)


def get_input_text_mi(premise, hypothesis):
    return ZERO_SHOT_CLASSIFIER_PROMPT.format(sentence=f"{premise} \n This imply that {hypothesis}.")


def get_input_text(premise, hypothesis):
    options_prefix = "OPTIONS:\n- "
    separator = "\n- "
    options_ = options_prefix + f"{separator}".join(["Entailment", "Contradiction"])
    return f"{premise} \n Question: Does this imply that {hypothesis}? {options_}"


def prepare(split='train.json', prompt_type='train', mode='normal'):
    if split == 'all.json':
        splits = ['train.json', 'dev.json']
    else:
        splits = [split]

    data_expanded = []
    for split in splits:
        data = json.load(open(f"Complete_dataset/{split}"))
        files = os.listdir("Complete_dataset/CT json/")
        files.remove(".DS_Store")

        files_data = {file[:-5]: json.load(open(f"Complete_dataset/CT json/{file}")) for file in files}

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
            if 'Label' in value:
                temp["label"] = value["Label"]

            if s_nctid is not None:
                secondary_evidence = files_data[s_nctid][section_id]
                temp["secondary_evidence"] = secondary_evidence

            data_expanded.append(temp)

    if mode == 'fast':
        data_expanded = data_expanded[:20]
    samples = []
    label2idx = {'Entailment': 0, 'Contradiction': 1}
    for idx, sample in enumerate(data_expanded):
        if 'label' in sample:
            if sample['label'] == 'Contradiction':
                label = 'No'
            if sample['label'] == 'Entailment':
                label = 'Yes'
        primary_evidence = "".join(sample['primary_evidence'])
        sentence = f"Primary trial evidence are {primary_evidence}"
        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary_evidence = "".join(sample['secondary_evidence'])
            sentence = f"{sentence} Secondary trial evidence are {secondary_evidence}"
        if prompt_type == 'train':
            input_text = TRAIN_PROMPT.format(premise=sentence, hypothesis=sample['statement'], label=label)
        if prompt_type == 'eval':
            input_text = EVAL_PROMPT.format(premise=sentence, hypothesis=sample['statement'])
        if prompt_type == 'train_base':
            input_text = TRAIN_PROMPT_BASE.format(premise=sentence, hypothesis=sample['statement'], label=label)
        if prompt_type == 'inference':
            input_text = get_input_text_p(sentence, sample['statement'], NEW_PROMPT)
        if 'label' in sample:
            temp = {"text": input_text, "label": sample['label'],
                    'id': idx, "label_cls": label2idx[sample['label']],
                    'id_': sample['id']}
        else:
            temp = {"text": input_text, 'id': idx, 'id_': sample['id']}
        samples.append(temp)

    def gen():
        for sample in samples:
            yield sample

    dataset = Dataset.from_generator(gen)

    return dataset


def prepare_bionli(split='train.csv', prompt_type='train'):
    samples = []
    substrings_to_replace = ["<el>", "<le>", "<re>", "<er>"]
    data = pd.read_csv(f'data/bionli/{split}')

    data = data.to_dict(orient='records')

    devd = pd.read_csv(f'data/bionli/dev.csv')

    data.extend(devd.to_dict(orient='records'))

    for sample in data:
        if sample['label_cat'] == 'pos':
            label = 'Yes'
        else:
            label = 'No'
        sentence = sample['supp_set']
        hypothesis = sample['conclusion']
        for substring in substrings_to_replace:
            hypothesis = hypothesis.replace(substring, '')

        if prompt_type == 'train':
            input_text = TRAIN_PROMPT.format(premise=sentence, hypothesis=hypothesis, label=label)
        if prompt_type == 'train_base':
            input_text = TRAIN_PROMPT_BASE.format(premise=sentence, hypothesis=hypothesis, label=label)
        if prompt_type == 'inference':
            input_text = get_input_text_p(sentence, sample['statement'], NEW_PROMPT)
        temp = {"text": input_text, "label": "Entailment" if label == 'Yes' else 'Contradiction'}
        samples.append(temp)

    def gen():
        for sample in samples:
            yield sample

    dataset = Dataset.from_generator(gen)
    ct_dataset = prepare(prompt_type=prompt_type)
    dataset = concatenate_datasets([ct_dataset, dataset])
    return dataset


def prepare_med_nli(split='mli_train_v1.jsonl', prompt_type='train'):
    base_path = 'data/med_nli/'
    data = []
    samples = []
    label2idx = {'Entailment': 0, 'Contradiction': 1}
    with open(f'{base_path}{split}', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)

    for idx, sample in enumerate(data):
        if sample['gold_label'] == 'entailment':
            sample['gold_label'] = 'Entailment'
            label = 'Yes'
        elif sample['gold_label'] == 'contradiction':
            sample['gold_label'] = 'Contradiction'
            label = 'No'
        else:
            continue
        sentence = sample['sentence1']
        hypothesis = sample['sentence2']

        if prompt_type == 'train':
            input_text = TRAIN_PROMPT.format(premise=sentence, hypothesis=hypothesis, label=label)
        if prompt_type == 'train_base':
            input_text = TRAIN_PROMPT_BASE.format(premise=sentence, hypothesis=hypothesis, label=label)
        if prompt_type == 'inference':
            input_text = get_input_text_p(sentence, hypothesis, NEW_PROMPT)
        if prompt_type == 'classification':
            input_text = get_input_text_p(sentence, hypothesis, CLASS_PROMPT)
        if prompt_type == 'few_shot':
            input_text = FEW_SHOT_PROMPT.format(premise=sentence, hypothesis=hypothesis, label=sample['gold_label'])
        temp = {"text": input_text, "label": sample['gold_label'], 'id': idx,
                "label_cls": label2idx[sample['gold_label']]}
        samples.append(temp)

    def gen():
        for sample in samples:
            yield sample

    dataset = Dataset.from_generator(gen)
    return dataset, samples