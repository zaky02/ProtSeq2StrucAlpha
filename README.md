# ProtSeq2StrucAlpha

ProtSeq2StrucAlpha is a deep learning framework designed to predict Foldseek structural tokens directly from protein amino acid sequences. By leveraging a transformer-based encoder-decoder architecture, this project aims to bridge the gap between protein sequences and their structural representations, facilitating rapid structural annotation and analysis.

## Overview

The core objective of ProtSeq2StrucAlpha is to translate protein sequences into their corresponding structural token sequences as defined by Foldseek. This approach enables:

- **Efficient Structural Annotation**: Rapid prediction of structural features without the need for computationally intensive methods.
- **Enhanced Protein Analysis**: Facilitates downstream tasks such as protein classification, function prediction, and interaction analysis.

## Repository Structure

- `bin/` – Executable scripts
- `data_preparation/` – Dataset preparation scripts
- `old_scripts/` – Archived legacy code
- `utils/` – Utility functions
- `config.json` – Model and training configuration
- `dataset.py` – Dataset loading and preprocessing
- `hyperparameter_tuning.py` – Hyperparameter optimization
- `model.py` – Transformer model definition
- `predict.py` – Prediction interface
- `sync_wandb.py` – Weights & Biases experiment syncing
- `tokenizer.py` – Protein sequence tokenization
- `train.py` – Main training script
- `train_cross.py` – Alternative training routine
- `ProtSeq2StrucAlpha_environment.yml` – Conda environment file

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Conda package manager

### Installation

```bash
git clone https://github.com/zaky02/ProtSeq2StrucAlpha.git
cd ProtSeq2StrucAlpha
conda env create -f ProtSeq2StrucAlpha_environment.yml
conda activate ProtSeq2StrucAlpha
```

## Usage

### Training the Model

To train the model with the default configuration:

```bash
python train.py --config CONFIG --dformat DFORMAT
```

### Making predictions

After training, generate predictions using:

```bash
python predict.py --input_file path_to_input_sequences.fasta --output_file predictions.txt
```

