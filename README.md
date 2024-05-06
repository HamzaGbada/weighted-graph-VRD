# Weighted-graph-VRD

This repository contains the official implementation of the paper titled [Multimodal weighted graph representation for information extraction from visually rich documents](https://www.sciencedirect.com/science/article/abs/pii/S0925231223013462).

## Abstract

The paper introduces a novel system for information extraction from visually rich documents (VRD) using a weighted graph representation. It aims to enhance the performance of information extraction tasks by capturing relationships between various VRD components. VRD is modeled as a weighted graph, encoding visual, textual, and spatial features of text regions as nodes and edges representing relationships between neighboring text regions. Information extraction from VRD is treated as a node classification task using graph convolutional networks. The approach is evaluated across diverse documents, including invoices and receipts, achieving performance levels equal to or exceeding robust baselines.

## Dependencies

- [DGL](https://www.dgl.ai/)
- [PyTorch](https://pytorch.org/)
- [Python](https://www.python.org/)
- [NetworkX](https://networkx.org/documentation/stable/tutorial.html)
- [OpenCV-Python](https://opencv.org/)

## Usage

### Graph Builder Command

To build a graph-based dataset, use the following command:

```shell script
$ python graph_builder.py -h   
```

This command creates a graph-based dataset for node classification for a specific dataset to extract entities from Visually Rich Documents.

Optional Arguments:
- `-d DATASET, --dataset DATASET`: Choose the dataset to use. Options are `FUNSD`, `SROIE`, `Wildreceipt` or `CORD`.
- `-t TRAIN, --train TRAIN`: Boolean to choose between the train or test dataset.

### Training Command

To train the model, use the following command:

```shell script
$ python train.py -h
```

This command trains the model on a selected dataset for node classification.

Arguments:
- `-d DATANAME, --dataname DATANAME`: Select the dataset for model training. Options are `FUNSD`, `SROIE`, `Wildreceipt` or `CORD`.
- `-p PATH, --path PATH`: Select the dataset path for model training.
- `-hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE`: GCN hidden size.
- `-hl HIDDEN_LAYERS, --hidden_layers HIDDEN_LAYERS`: Number of GCN hidden layers.
- `-lr LEARNING_RATE, --learning_rate LEARNING_RATE`: The learning rate.
- `-e EPOCHS, --epochs EPOCHS`: The number of epochs.

## Acknowledgments
We acknowledge the contributions of the authors of the paper and the developers of the libraries used in this project.




