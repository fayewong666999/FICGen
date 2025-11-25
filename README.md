# Implementation of FICGen (ICCV 2025)

FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation
<a href='https://arxiv.org/abs/2509.01107'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
> *Wenzhuang Wang, [Yifan Zhao](https://zhao1f.github.io/), Mingcan Ma, Ming Liu,  Zhonglin Jiang, Yong Chen, [Jia Li](http://cvteam.net/members/lijia/upload/index.html)
> <br>
> State Key Laboratory of Virtual Reality Technology and Systems, SCSE&QRI, Beihang University, Geely Automobile Research Institute (Ningbo) Co., Ltd

<img src="assets/2.jpg" width="800">

## Features
* **Motivation:** The architecture of diffusion models is transitioning from Unet-based to DiT (Diffusion Transformer). However, the DiT ecosystem lacks mature plugin support and faces challenges such as efficiency bottlenecks, conflicts in multi-condition coordination, and insufficient model adaptability.
* **Contribution:** We propose EasyControl, an efficient and flexible unified conditional DiT framework. By incorporating a lightweight Condition Injection LoRA module, a Position-Aware Training Paradigm, and a combination of Causal Attention mechanisms with KV Cache technology, we significantly enhance **model compatibility** (enabling plug-and-play functionality and style lossless control), **generation flexibility** (supporting multiple resolutions, aspect ratios, and multi-condition combinations), and **inference efficiency**.
<img src='assets/method.jpg'>

## News
- **2025-04-11**: 🔥🔥🔥 Training code have been released. Recommanded Hardware: at least 1x NVIDIA H100/H800/A100, GPUs Memory: ~80GB GPU memory.
- **2025-04-09**: ⭐️ The codes for the simple API have been released. If you wish to run the models on your personal machines, head over to the simple_api branch to access the relevant resources.

- **2025-04-07**: 🔥 Thanks to the great work by [CFG-Zero*](https://github.com/WeichenFan/CFG-Zero-star) team, EasyControl is now integrated with CFG-Zero*!! With just a few lines of code, you can boost image fidelity and controllability!! You can download the modified code from [this link](https://github.com/WeichenFan/CFG-Zero-star/blob/main/models/easycontrol/infer.py) and try it.
