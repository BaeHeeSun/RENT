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
