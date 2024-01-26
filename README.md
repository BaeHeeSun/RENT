# RENT (Dirichlet-based Per-Sample Weighting by Transition Matrix for Noisy Label Learning, ICLR 2024)

Official PyTorch implementation of
[**"Dirichlet-based Per-Sample Weighting by Transition Matrix for Noisy Label Learning"**](https://openreview.net/forum?id=A4mJuFRMN8) (ICLR 2024) by
[HeeSun Bae](https://sites.google.com/view/baeheesun),
[Seungjae Shin](https://sites.google.com/view/seungjae-shin),
[Byeonghu Na](https://sites.google.com/view/byeonghu-na),
and [Il-Chul Moon](https://aailab.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=3).

## Overview

For learning with noisy labels, we propose a good transition matrix utilization is crucial for training a model, and suggest a new utilization method based on resampling, coined **RENT**.

We first demonstrate current utilizations can have potential limitations for implementation. As an extension to Reweighting, we suggest the **D**irichlet distribution-based per-sample **W**eight **S**ampling (DWS) framework, and **compare reweighting and resampling** under DWS framework. 

We propose **RENT**, a **RE**sampling method with **N**oise **T**ransition matrix. Empirically, RENT consistently outperforms existing transition matrix utilization methods, which includes reweighting, on various benchmark datasets.

## Setup

Install required libraries.

```
pip install -r requirements.txt
```

For running the code, please refer to the README.md inside each folder, **DWS** and **RENT**.

## Dataset

If you run the code, the noisy label of the benchmark dataset (e.g. CIFAR10) will be generated automatically.
However, we also provide the noisy label dataset we used for our experiments [https://drive.google.com/drive/u/0/folders/1UHVUev0KLqzbpR6kjyn0HtHpUvC0uQps](https://drive.google.com/drive/u/0/folders/1UHVUev0KLqzbpR6kjyn0HtHpUvC0uQps).

It includes:
* `clean` : no noise (same as the original dataset)
* `sym` : symmetric noise (or uniforma noise). Note that the noisy label ratio represents the ratio the label is "different" from the original label.
* `asym` : asymmetric noise which refers to the situation when the labels are flipped according to the pre-defined class similarity relationship.
* `real_xxx` : realistic noise from [Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](http://www.noisylabels.com/).

Numbers represent noise ratio. For real noise, since the noise ratio is already determined, we put 1.0.

## Performance
Here we show the main performance (metric: test accuracy, mean over 5 replications) table. 

The transition matrix is estimated following [this](https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf). However, check also for the results with different transition matrix estimation methods.

### CIFAR10

Noise | SN20% | SN50% | ASN20% | ASN40%
--- | --- | --- | --- |--- 
CE|73.4|46.6|78.4|69.7
w/FL | 73.8 | 58.8 | 79.2 | 74.2
w/RW | 74.5 | 62.6 | 79.6 | 73.1
**w/RENT** | **78.7** | **69.0** | **82.0** | **77.8**

### CIFAR100

Noise | SN20% | SN50% | ASN20% | ASN40%
--- | --- | --- | --- |--- 
CE|33.7|18.5|36.9|27.3
w/FL | 30.7 | 15.5 | 34.2 | 25.8
w/RW | 37.2 | 23.5 | 27.2 | 27.3
**w/RENT** | **38.9** | **28.9** | **38.4** | **30.4**

### Real Noise

Noise from **Agree** to **Worse** are from CIFAR-10N.

Noise | Agree | Ran1 | Ran2 | Rand3 | Worse| Clothing1M |
--- | --- | --- | --- |--- | --- |---
CE|80.8|75.6|75.3|75.6|60.4|66.9
w/FL|79.5|76.1|76.4|76.0|64.5|67.1
w/RW|80.7|75.8|76.0|75.8|63.9|66.8
**w/RENT**|**80.8**|**77.7**|**77.5**|**77.2**|**68.0**|**68.2**

**Please refer to the paper also for more experimental results, including:**
* Analytical results: e.g. The impact of $\alpha$, Noise injection impact, Confidence comparison, Resampled dataset quality
* Performance Comparison with regard to the noise ratio
* Impact of Transition matrix estimation
* Comparison with different noisy label learning methods
* General Performance under various experimental settings, e.g. Optimizer, Network architecture.
* When the dataset is class imbalanced and noisy
* Time complexity
