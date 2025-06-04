# Executable Dependencies Recognition Based on Control Flow Graph Analysis and GNN

[![ru](https://img.shields.io/badge/lang-ru-blue.svg)](https://github.com/lixxteq/gembin/blob/master/README.md)
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/lixxteq/gembin/blob/master/README_EN.md)

## Requirements

- Linux / Cygwin toolkit for Windows
- Python 3.10-3.12
- radare2 > 5.8.x
- IDA Pro > 7.7 (required when using IDAPython driver for feature extraction)

## Usage

### Feature extraction

```sh
python application/feature_extractor.py <executable path> -f <function to extract> -o <output ACFG file path>
```

### Similarity inference

```sh
python application/similarity_mp.py <lib ACFG file> <target ACFG file>
```

### Model training

Training requires dataset with executable files in `data` directory.
Naming of files in dataset should be configured in `config.py` and `train.py`.

```sh
python train.py --log_path <log path> --save_path <optional model path>
```

## Visualization of vector embeddings and similarity inference

> Install optional dependencies in `requirements.optional.txt` to visualize vector embeddings in Gradio web interface.

### Raw (precomputed) vectors visualization

Precomputing 64-dimensional vectors from ACFG JSON files:

```sh
python generate_emb.py --output_tsv embeddings.tsv demo_files/func1.json demo_files/func2.json ...
```

Visualization in 3D space (using PCA reduction) and cosine similarity / Euclidean distance calculation:

```sh
python visualize_3d.py
```

> [!NOTE]
> PCA reduction to 3-dimensional space requires at least 3 raw vectors as input.

> [!NOTE]
> Cosine similarity / Euclidean distance are being calculated on raw vectors and do not use pretrained model for similarirty inference. To perform real similarity inference, refer to next step.

### Visualization of similarity inference

Similarity inference on all function's ACFGs for each input JSON ACFG file:

```sh
python visualize_inference.py demo_files/func1.json demo_files/func2.json ...
```

## WIP

- IDA Pro IDAPython driver usage description