# TopoDerma
Code for the paper <br />

Our project explores the application of topological deep learning to improve skin cancer classification across 4 diverse datasets. Through shared self-attention, CLS tokens are able to dynamically select and aggregate the best features from the topology and vision transformer inputs. <br /> 

## Table of Contents
* [Installation](#installation)
* [Data](#data)
* [Results](#results)
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
* [DermaMNIST](https://medmnist.com/)
* [DDI](https://ddi-dataset.github.io/)
* [PH2](https://www.fc.up.pt/addi/ph2%20database.html)
* [MED-NODE](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/) <br />

## Results
The detailed results can be found in our paper. To summarize, our topology enhanced models improve existing deep learning model baselines. 

## Model Architecture


## Acknowledgements 
* We would like to thank the authors of the [datasets](#data) and the creators of [timm](https://timm.fast.ai/) for their time and effort in developing these valuable resources
