# Fine-Tuning ESM-3 on Soluble High-Enrichment DARPin Sequences

This repository provides a comprehensive framework for fine-tuning the ESM-3 protein language model on a curated dataset of Designed Ankyrin Repeat Proteins (DARPins). Below, you'll find an overview of ESM, the fine-tuning strategies employed, and instructions on how to run the code.

---

## What is ESM?

ESM (Evolutionary Scale Modeling) is a family of large-scale language models for proteins. This repository uses **ESM-3**, a frontier generative model for biology that can reason across three fundamental properties of proteins: sequence, structure, and function.

- **ESM-3**: A multimodal generative model that can be prompted with partial sequence, structure, and function information to generate novel protein sequences.
- **ESM C**: A model focused on creating high-quality representations of the underlying biology of proteins, ideal for embedding and prediction tasks.

This repository focuses on fine-tuning ESM-3 to enhance its capabilities for generating stable and high-affinity DARPin sequences.

---

## Fine-Tuning Strategies

We employ several fine-tuning strategies to adapt ESM-3 to the specific characteristics of DARPins.

### 1. Full Model Fine-Tuning

This approach involves updating all the parameters of the ESM-3 model. It is the most comprehensive but also the most computationally expensive method.

- **Script**: `ESM3_fullmodel_finetune.py`
- **Use Case**: When you want to adapt the entire model to a new data distribution and have sufficient computational resources.

### 2. Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods allow for efficient adaptation of large models with fewer trainable parameters, reducing computational overhead. We have implemented two popular PEFT techniques:

#### Low-Rank Adaptation (LoRA)

LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This significantly reduces the number of trainable parameters.

#### Weight-Decomposed Low-Rank Adaptation (DoRA)

DoRA is a variant of LoRA that decomposes the pre-trained weights into two components, magnitude and direction, and applies LoRA to the direction component. This can lead to better performance and more stable training.

- **Script**: `esm3_lora.py`
- **Configuration**: The `LoraConfig` in the script allows you to switch between LoRA and DoRA and configure the following key parameters:
    - `r`: The rank of the update matrices.
    - `lora_alpha`: The scaling factor for the LoRA updates.
    - `use_dora`: Set to `True` to use DoRA or `False` for standard LoRA.
    - `target_modules`: The modules to which the LoRA updates are applied.

---

## Project Objective

DARPins are engineered protein scaffolds with high specificity and stability, making them valuable for therapeutics and diagnostics. By fine-tuning ESM-3 with highly enriched, soluble DARPins, we aim to:

- Improve downstream property prediction (e.g., solubility, affinity).
- Enable the generation of de novo DARPin-like sequences with better developability.
- Benchmark model performance against held-out test sets and known DARPin libraries.

---

## Running the DARPin–ESM-3 Pipeline

This section outlines the steps to prepare DARPin sequences and fine-tune the ESM-3 model.

### Step 0: Installation

Create a conda environment and activate it:

```bash
conda create -n esm3 python=3.9 -y
conda activate esm3
```

Then, install the necessary packages and clone the repository:

```bash
pip install esm
git clone https://github.com/tau-team/esm3-darpins.git
cd Scripts
```

### Step 1: Generate FASTA from Scored Sequences

Run the FASTA generation script to select the top sequences by score:

```bash
python3 make_fasta.py
```

This script reads a CSV file with `Sequence` and `Score` columns, sorts by score, and writes the top sequences to `fasta/darpin.fasta`.

### Step 2: Train the Model

Choose one of the fine-tuning strategies:

- **LoRA or DoRA**:
  ```bash
  python3 esm3_lora.py
  ```
- **Full Model Fine-tuning**:
  ```bash
  python3 ESM3_fullmodel_finetune.py
  ```

### Step 3: Run Inference

To perform inference on the fine-tuned model, run:

```bash
python3 esm3_model_inference.py
```

---

## Dataset

The training dataset includes curated DARPin sequences with:

- **Solubility**: > threshold (experimentally validated or predicted)
- **Enrichment Score**: > threshold (based on phage or yeast display rounds)

Supported formats: `FASTA`, `CSV` (with metadata)
