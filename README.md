# RobustLLM
Code Repository for the SemEval-2024 Task 2: Safe Biomedical Natural 
Language Inference for Clinical Trials.


## Usage

```bash
# Create a conda environment
conda create -n rllm python=3.9
conda activate rllm

# Install required packages
pip install -r requirements.txt
```

## Train

```bash
python src/train.py -config configs/train.yaml
```

## Inference

```bash
python src/inference.py
```

## Hyperparameter Tuning

```bash
python src/hpt.py
```

## Data Perturbations

There are two types of perturbations that we introduced to NLI4CT dataset: acronym 
    and numerical. Acronym based perturbations can be replicated using 
[nlict_data_acronym_perturbations.ipynb](notebooks/nlict_data_acronym_perturbations.ipynb) 
and numerical based perturbations can be replicated using [nlict_data_numerical_perturbations.ipynb](notebooks/nlict_data_numerical_perturbations.ipynb).