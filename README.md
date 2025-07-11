# TopoDerma
Code for the paper 'Topology-Aware Deep Models for Skin Lesion Classification' <br />

Our project explores the application of topological deep learning to improve skin cancer classification across 4 diverse datasets. Through shared self-attention, CLS tokens are able to dynamically select and aggregate the best features from the context-aware persistent homology tokens and vision transformer tokens. <br /> 

## Table of Contents
* [Installation](#installation)
* [Data](#data)
* [Results](#results)
* [Model Architecture](#model-architecture)
* [Acknowledgements](#acknowledgements)


## Installation
+ Python ≥ 3.9

+ PyTorch ≥ 2.0

+ timm ≥ 0.9
```
pip install torch>=2.0 timm>=0.9
```

## Data
The links to the original datasets can be found below: <br />
* [DermaMNIST](https://medmnist.com/) (ISIC 2018)
* [DDI](https://ddi-dataset.github.io/)
* [PH2](https://www.fc.up.pt/addi/ph2%20database.html)
* [MED-NODE](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/) <br />

## Results
The detailed results can be found in our paper. To summarize, our topology enhanced models improve existing deep learning model baselines. 

## Model Architecture
![TopoDerma2](https://github.com/user-attachments/assets/a5f68dd1-2acb-4f70-8cb6-79283ee983dd)




## Acknowledgements 
* We would like to thank the authors of the [datasets](#data) and the creators of [timm](https://timm.fast.ai/) for their time and effort in developing these valuable resources
