# HiHPO
HiHPO: Multimodal Hierarchical Graph Learning for Predicting Missing Protein-Phenotype Associations

## Introduction
HiHPO, a novel framework designed to predict missing HPO annotations for proteins by integrating multimodal protein data, including PPI networks, gene expression profiles, and ESM sequence embeddings, with a hierarchy-aware contrastive learning module.

<img src="data/framwork.png" style="zoom:33%;" />

## Requirements

Our model is implemented by Python 3.8 with Pytorch 2.3.1 and Pytorch-geometric 2.5.3, and run on Nvidia GPU with CUDA 12.4.

## Usage

Please download the data (https://drive.google.com/drive/folders/1a1uRm8cB9Yjcvx7tAUhK0UnsOPAL2ZMn?usp=share_link) and put them into the `data` directory,

Run the following command to train the model:

```
python main.py
```