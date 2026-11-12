# TopoDerma

Topological features for skin lesion classification with CNNs and vision transformers.

**[Topology-Aware Deep Models for Skin Lesion Classification](https://workshop.isic-archive.com/2025/paper_chakraborty.pdf)**

Sayoni Chakraborty\*, Philmore Koung\*, and Baris Coskunuzer

\*Equal contribution · University of Texas at Dallas

[ISIC Workshop at MICCAI 2025](https://workshop.isic-archive.com/2025/) · **Honorable Mention Award**

[Overview](#overview) · [Installation](#installation) · [Data](#data) · [Usage](#usage) · [Experiments](#experiments) · [Citation](#citation)

## Overview

TopoDerma combines persistent-homology descriptors with image representations. Cubical persistence produces Betti curves for connected components and loops; these features can be fused with CNN embeddings or encoded as tokens alongside image tokens in a shared transformer.

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

## Installation

Clone the repository and create a Python 3.10 environment:

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

Choose framework builds for your operating system and accelerator. Both model builders use pretrained image weights by default. Set `pretrained=False` for `TopoViT` or `weights=None` for the CNN builder when training from scratch.

## Data

Obtain images and labels from the original providers and follow their access terms:

| Dataset | Tasks in the paper | Source |
| --- | --- | --- |
| ISIC 2018 / DermaMNIST | Seven-class and binary classification | [MedMNIST](https://medmnist.com/) |
| Diverse Dermatology Images (DDI) | Binary classification | [DDI](https://ddi-dataset.github.io/) |
| PH2 | Three-class classification | [PH2 database](https://www.fc.up.pt/addi/ph2%20database.html) |
| MED-NODE | Binary classification | [MED-NODE database](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/) |

Keep image, feature, and label rows in the same order. Compute topology features from the original OpenCV-style **BGR** images before applying backbone-specific image preprocessing.

## Usage

Run the following Python examples from the repository root.

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

**Feature update:** Grayscale H1 now measures loops instead of duplicating H0. Regenerate features and retrain models created with the earlier extractor.

The extractor uses cubical persistence with coefficient field 3 and 50-bin Betti curves. It fits the curve sampling separately for each image/channel/homology calculation. The returned DataFrame repeats column labels `0` through `49` for each block; use positional column order when exporting or loading features. Set `n_jobs=1` in `generate_betti_vectors` for bounded preprocessing; the default remains `-1`.

### 2. Build the vision transformer

```python
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
)
```

`TopoViT` accepts images of shape `(batch, 3, 224, 224)` and features of shape `(batch, 400)` or `(batch, 25, 16)` with this configuration. It uses the `swin_tiny_patch4_window7_224` image backbone and returns class logits. Apply the backbone's input normalization to the image branch.

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

## Experiments

The [paper](https://workshop.isic-archive.com/2025/paper_chakraborty.pdf) describes the experimental protocol and reports results in Tables 2-4.

For new experiments, record the dataset version, split membership, image preprocessing, feature column order, random seeds, framework versions, and selected checkpoint. Preserve alignment between images and topology features throughout batching and splitting.

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
