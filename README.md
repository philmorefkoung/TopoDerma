# TopoDerma

Topological features for skin lesion classification with CNNs and vision transformers.

**[Topology-Aware Deep Models for Skin Lesion Classification](https://workshop.isic-archive.com/2025/paper_chakraborty.pdf)**

Sayoni Chakraborty\*, Philmore Koung\*, and Baris Coskunuzer

\*Equal contribution · University of Texas at Dallas

[ISIC Workshop at MICCAI 2025](https://workshop.isic-archive.com/2025/) · **Honorable Mention Award**

[Overview](#overview) · [Installation](#installation) · [Data](#data) · [Usage](#usage) · [Reproducibility](#reproducibility) · [Citation](#citation)

## Overview

TopoDerma combines persistent-homology descriptors with image representations. Cubical persistence produces Betti curves for connected components and loops; these features can be fused with CNN embeddings or encoded as tokens alongside image tokens in a shared transformer.

This repository provides feature extraction and model-building components. Dataset loaders, training/evaluation drivers, trained checkpoints, and experiment configurations are not included.

![TopoDerma architecture: image and persistent-homology branches feed a shared classifier](https://github.com/user-attachments/assets/a5f68dd1-2acb-4f70-8cb6-79283ee983dd)

### Repository guide

| File | Purpose |
| --- | --- |
| [BettiVectorization.py](BettiVectorization.py) | Extract 400-dimensional Betti vectors from BGR images using `generate_betti_vectors`. |
| [TopoViT.py](TopoViT.py) | Build a PyTorch `TopoViT` with a pretrained Swin-Tiny image branch and a persistent-homology token branch. |
| [topo_cnn.py](topo_cnn.py) | Build a TensorFlow/Keras CNN + topology classifier with DenseNet121, MobileNetV2, or InceptionV3. |
| [requirements.txt](requirements.txt) | Dependencies for feature extraction. |
| [requirements-vit.txt](requirements-vit.txt) / [requirements-cnn.txt](requirements-cnn.txt) | Dependencies for the chosen model family, including feature extraction. |
| [CITATION.cff](CITATION.cff) | Machine-readable citation metadata. |
| [tests/test_betti_vectorization.py](tests/test_betti_vectorization.py) | Regression for the grayscale H1 block using a synthetic ring and real persistent homology. |
| [scripts/smoke_train_test.py](scripts/smoke_train_test.py) | Bounded synthetic training and held-out evaluation for each model family. |

## Installation

Clone the repository and create a dedicated environment. Python 3.10 is a suggested starting point; the repository does not provide a locked environment from the paper experiments.

```bash
git clone https://github.com/philmorefkoung/TopoDerma.git
cd TopoDerma
conda create -n topoderma python=3.10 -y
conda activate topoderma
```

Install only the components you need:

```bash
# Feature extraction
python -m pip install -r requirements.txt

# Feature extraction + PyTorch vision transformer
python -m pip install -r requirements-vit.txt

# Feature extraction + TensorFlow CNN
python -m pip install -r requirements-cnn.txt
```

These files describe imported packages, not a validated lockfile. Choose framework builds appropriate for your operating system and accelerator. Both model builders request pretrained image weights by default, which may be downloaded on first use. Use `pretrained=False` for `TopoViT` or `weights=None` for the CNN builder to initialize without downloading weights.

## Data

Obtain images and labels from the original providers and follow their access terms:

| Dataset | Tasks in the paper | Source |
| --- | --- | --- |
| ISIC 2018 / DermaMNIST | Seven-class and binary classification | [MedMNIST](https://medmnist.com/) |
| Diverse Dermatology Images (DDI) | Binary classification | [DDI](https://ddi-dataset.github.io/) |
| PH2 | Three-class classification | [PH2 database](https://www.fc.up.pt/addi/ph2%20database.html) |
| MED-NODE | Binary classification | [MED-NODE database](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/) |

Keep image, feature, and label rows in the same order. The extractor expects OpenCV-style **BGR** images; image backbones require their own preprocessing. Do not feed normalized backbone tensors directly into the Betti extractor.

## Usage

Run the following Python examples from the repository root. They demonstrate the released interfaces; they are not training or paper-reproduction scripts.

### 1. Extract Betti vectors

```python
import cv2
from BettiVectorization import generate_betti_vectors

image = cv2.imread("path/to/lesion.jpg")  # BGR uint8 image
if image is None:
    raise FileNotFoundError("Could not read the image")
image = cv2.resize(image, (224, 224))

features = generate_betti_vectors([image])
assert features.shape == (1, 400)
features.to_csv("betti_features.csv", index=False)
```

Each row contains eight consecutive blocks of 50 values, in this order:

```text
gray H0 | gray H1 | red H0 | red H1 | blue H0 | blue H1 | green H0 | green H1
```

**Feature correction:** The grayscale H1 block now uses `CP1` to measure loops. Earlier code called `CP0` for this block and duplicated grayscale H0. Regenerate cached features and retrain models when adopting this correction; earlier feature files and results are not interchangeable with corrected ones.

The extractor uses cubical persistence with coefficient field 3 and 50-bin Betti curves. It fits the curve sampling separately for each image/channel/homology calculation. The returned DataFrame repeats column labels `0` through `49` for each block; use positional column order when exporting or loading features. Set `n_jobs=1` in `generate_betti_vectors` for bounded preprocessing; the default remains `-1`.

### 2. Build the vision transformer

```python
import torch
from TopoViT import TopoViT

model = TopoViT(
    num_classes=7,
    ph_in_dim=16,
    ph_token_len=25,  # 25 x 16 = 400 features
    ph_embed_dim=128,
    ph_num_heads=4,
    tf_num_heads=8,
    tf_layers=6,
    dropout=0.3,
    pretrained=False,  # Random initialization for this synthetic interface check.
)
model.eval()

# Synthetic tensors for an interface check, not a prediction on a real lesion.
images = torch.zeros(1, 3, 224, 224)
ph_features = torch.zeros(1, 400)
with torch.no_grad():
    logits = model(images, ph_features)
assert logits.shape == (1, 7)
```

`TopoViT` accepts flat features of shape `(batch, 400)` or tokenized features of shape `(batch, 25, 16)` with this configuration. It returns logits. The released image backbone is `swin_tiny_patch4_window7_224`; other backbones evaluated in the paper are not selected by an argument in this implementation. Use the selected backbone's input normalization for real images.

### 3. Build a CNN + topology classifier

```python
from topo_cnn import build_cnn_tda_model

model = build_cnn_tda_model(
    input_shape_img=(224, 224, 3),
    tda_dim=400,
    num_classes=7,
    base_model_name="DenseNet121",
)
model.summary()
```

The Keras model accepts `[images, betti_features]`, with shapes `(batch, 224, 224, 3)` and `(batch, 400)`. It freezes the ImageNet backbone and compiles a softmax classifier with sparse categorical cross-entropy. Labels must be integer class indices. Apply the selected Keras backbone's `preprocess_input` to the image branch; compute the topology features from the original BGR images first.

## Reproducibility

The [paper](https://workshop.isic-archive.com/2025/paper_chakraborty.pdf) describes the experimental protocol and reports results in Tables 2-4. Reproducing those results requires the original data partitions, preprocessing, training settings, and evaluation code in addition to these components. In particular, the released CNN builder uses a softmax/sparse-categorical setup and should not be assumed to implement the paper's separate binary-training recipe.

For new experiments, record the dataset version, split membership, image preprocessing, feature column order, random seeds, framework versions, and selected checkpoint. Preserve alignment between images and topology features throughout batching and splitting.

## Small validation runs

With the relevant requirements installed, run the persistence regression and each model check from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_train_test.py --model vit --steps 2
python scripts/smoke_train_test.py --model cnn --steps 2
```

The regression checks that a synthetic dark ring has a nonzero grayscale H1 curve, distinct from H0 and equal to the identical color-channel H1 curves. The model checks extract real Betti features from four training images and two separate held-out images, take two optimizer steps with batches of two, and evaluate only after training. They assert finite losses and parameter updates and print a JSON summary. Both use random initialization without weight downloads. The ViT check trains the image and topology branches with one fusion layer; the CNN check trains a MobileNetV2-based classifier head with its backbone frozen, as in the builder. These tiny synthetic checks establish execution, not scientific accuracy or reproduction of the paper.

Verified on Windows CPU on **November 12, 2026**, with Python 3.10.20, NumPy 1.26.4, giotto-tda 0.6.2, scikit-learn 1.3.2, and OpenCV 4.10. The ViT run used PyTorch 2.11, torchvision 0.26, and timm 0.9.16; the CNN run used TensorFlow 2.18 and Keras 3.12.4. The real persistence regression passed and rejected the earlier grayscale implementation. Both model checks completed two training steps and held-out evaluation with finite losses and confirmed parameter updates. This records the tested configuration; it is not a locked environment or a guarantee across other package versions.

## Citation

If you use this work, please cite the paper. The entry below identifies the workshop manuscript linked above.

```bibtex
@inproceedings{chakraborty2025topoderma,
  title     = {Topology-Aware Deep Models for Skin Lesion Classification},
  author    = {Chakraborty, Sayoni and Koung, Philmore and Coskunuzer, Baris},
  booktitle = {Tenth ISIC Skin Image Analysis Workshop at MICCAI 2025},
  year      = {2025},
  url       = {https://workshop.isic-archive.com/2025/paper_chakraborty.pdf}
}
```

## Acknowledgements

We thank the dataset authors and the contributors to giotto-tda, OpenCV, PyTorch, timm, and TensorFlow/Keras for the resources used by this project.

## License

A software license has not yet been specified in this repository. Contact the maintainers to clarify reuse terms.
