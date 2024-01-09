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

TRAIN_PROMPT_BASE = """{premise}
Question: Does this imply that '{hypothesis}' ? Answer Yes or No.
### Answer: {label}
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


def prepare(split='train.json', prompt_type='train'):
    data = json.load(open(f"data/{split}"))
    files = os.listdir("data/CT json/")
    # files.remove(".DS_Store")

    files_data = {file[:-5]: json.load(open(f"data/CT json/{file}")) for file in files}

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
        temp["label"] = value["Label"]

        if s_nctid is not None:
            secondary_evidence = files_data[s_nctid][section_id]
            temp["secondary_evidence"] = secondary_evidence

        data_expanded.append(temp)

    samples = []
    for sample in data_expanded:
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
        if prompt_type == 'train_base':
            input_text = TRAIN_PROMPT_BASE.format(premise=sentence, hypothesis=sample['statement'], label=label)
        if prompt_type == 'inference':
            input_text = get_input_text_p(sentence, sample['statement'], NEW_PROMPT)
        temp = {"text": input_text, "label": sample['label']}
        samples.append(temp)

    def gen():
        for sample in samples:
            yield sample

    dataset = Dataset.from_generator(gen)

    return dataset
