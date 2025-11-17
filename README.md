# Fine-Tuning ESM-3 on Soluble High-Enrichment DARPin Sequences

This repository provides a comprehensive framework for fine-tuning the ESM-3 protein language model on a curated dataset of Designed Ankyrin Repeat Proteins (DARPins). Below, you'll find an overview of ESM, the fine-tuning strategies employed, and instructions on how to run the code.
<img width="1080" height="1080" alt="674f9301d43c2d67f061b290_667a5bb780d0ada7dc37d1c0_image%20(1)" src="https://github.com/user-attachments/assets/1075c8fb-eff2-4ecc-b75b-cabe7fe683c1" />

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

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry:**
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/Zahid8/ESM3-Adapters.git
    cd ESM3-Adapters
    poetry install
    ```

### Step 1: Generate FASTA from Scored Sequences

Run the FASTA generation script to select the top sequences by score:

```bash
poetry run python Scripts/make_fasta.py
```

This script reads a CSV file with `Sequence` and `Score` columns, sorts by score, and writes the top sequences to `fasta/darpin.fasta`.

### Step 2: Train the Model

Choose one of the fine-tuning strategies. The configuration for each script is managed by [Hydra](https://hydra.cc/). You can override the default parameters from the command line.

- **LoRA or DoRA**:
  ```bash
  poetry run python Scripts/esm3_lora.py
  ```
  To change the LoRA rank, for example:
  ```bash
  poetry run python Scripts/esm3_lora.py lora.r=16
  ```

- **Full Model Fine-tuning**:
  ```bash
  poetry run python Scripts/ESM3_fullmodel_finetune.py
  ```

### Step 3: Run Inference

To perform inference on the fine-tuned model, run:

```bash
poetry run python Scripts/esm3_inference.py
```

---

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. The configuration files are located in the `conf/` directory.

You can override any configuration parameter from the command line. For example, to change the learning rate for the LoRA fine-tuning script, you can run:

```bash
poetry run python Scripts/esm3_lora.py learning_rate=1e-5
```

---

## Dataset

The training dataset includes curated DARPin sequences with:

- **Solubility**: > threshold (experimentally validated or predicted)
- **Enrichment Score**: > threshold (based on phage or yeast display rounds)

Supported formats: `FASTA`, `CSV` (with metadata)
