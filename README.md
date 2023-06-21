# IM-Loss: Information Maximization Loss for Spiking Neural Networks

Official simplified implementation of IM-Loss.

## Introduction

The forward-passing spike quantization will cause information loss and accuracy degradation. To deal with this problem, the Information maximization loss (IM-Loss) that aims at maximizing the information flow in the SNN is proposed in the paper.

### Dataset

The dataset will be download automatically.

## Get Started

```
cd imloss
python main_train.py --spike --step 1 --distribution
```

## Citation

```bash
@inproceedings{
guo2022imloss,
title={{IM}-Loss: Information Maximization Loss for Spiking Neural Networks},
author={Yufei Guo and Yuanpei Chen and Liwen Zhang and Xiaode Liu and Yinglei Wang and Xuhui Huang and Zhe Ma},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022}
}
```
