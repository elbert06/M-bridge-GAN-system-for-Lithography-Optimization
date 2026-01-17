# M-Bridge GAN System for Lithography Optimization 🔬

[![Copyright](https://img.shields.io/badge/Copyright-C--2025--060029-blue.svg)](https://www.copyright.or.kr/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository contains the official source code for **"M-Bridge GAN System"**, a lightweight AI model designed to optimize **EUV (Extreme Ultraviolet) Lithography** processes.
Implemented using **TensorFlow/Keras**.

This project was registered with the **Korea Copyright Commission (No. C-2025-060029)**.

---
## 🚀 How to Run

1. **Download Dataset**
   - This project uses the **WM-811K** wafer map dataset.
   - Download the dataset from [Kaggle WM-811K Link](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) (or your source link).
   - Save the `WM811K.pkl` file into the `data/` directory.

   ```bash
   # Directory Structure
   M-Bridge-GAN/
   ├── data/
   │   └── WM811K.pkl  <-- Place the file here
   ├── main.py
   └── ...
## 📌 Abstract
The EUV lithography process often suffers from **yield instability** and high computational costs for defect detection. Traditional GAN models face challenges such as unstable training (mode collapse) and excessive resource consumption.

To solve this, I propose the **M-Bridge GAN**, which introduces a **'Mediator' module** between the Generator and Discriminator. This architecture significantly improves training stability and enables efficient inference on **edge devices** with limited hardware resources.

## ✨ Key Features
* **M-Bridge Architecture:** Introduces a Mediator to stabilize the adversarial training process.
* **Optimization Stability:** Maintains a stable Loss range of **0.4 ~ 0.6**, preventing mode collapse.
* **Lightweight Design:** Composed of approx. **5.6M parameters**, optimized for edge computing via TensorFlow.
* **Validated Performance:** Tested on **WM-811K** semiconductor wafer map dataset.

## 📊 Results
Comparison between Basic GAN and M-Bridge GAN (Proposed):

| Model | Training Stability | Loss Fluctuation |
| :--- | :--- | :--- |
| Basic GAN | Unstable | High |
| **M-Bridge GAN** | **Stable** | **Low (0.4~0.6)** |

> *The M-Bridge model demonstrates superior initial training stability compared to the baseline model.*

## 📜 Copyright Notice
**Registration Number:** C-2025-060029  
**Title:** M-Bridge GAN System for Lithography Optimization  
**Date of Registration:** Dec 29, 2025  
**Author:** Dohun Kim  

⚠️ *This code is protected by copyright law. Unauthorized reproduction or commercial use is prohibited without permission.*

## 📬 Contact
If you have any questions, please contact:  
**Dohun Kim** (Undergraduate, Dept. of Electrical Engineering, Inha University)  
📧 elbert06@hanmail.net