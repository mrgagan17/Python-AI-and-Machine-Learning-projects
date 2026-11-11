# 👤 Generative Adversarial Network (GAN) Face Synthesis

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

A custom Deep Convolutional Generative Adversarial Network (DCGAN) implementation designed to synthesize human faces from mathematical noise by pitting a Generator and Discriminator against each other in a zero-sum minimax game.

## 📸 Output Demo
*The Generator's output after initial CPU training. The neural network successfully captures spatial facial geometry and fundamental pixel distribution, demonstrating the architecture's readiness for scaled GPU training.*

![GAN Initial Training Output](output.jpg)

## 🚀 Key Features
*   **Dual Network Architecture:** Contains a pure PyTorch implementation (`gan_architecture.py`) featuring a `Generator` (ConvTranspose2d upsampling) and a `Discriminator` (Conv2d downsampling).
*   **Custom Training Engine:** Includes `train_gan.py`, a dedicated script to parse image datasets (like CelebA or LFW), calculate Binary Cross Entropy (BCE) loss, and optimize weights using Adam until the Generator successfully fools the Discriminator.
*   **Latent Space Decoding:** Dynamically generates random 100-dimensional noise vectors and processes them through batch normalization and ReLU activations to map latent coordinates to RGB pixel space.
*   **Production Deployment:** A dark-themed, interactive Streamlit dashboard allowing users to trigger the Generator on the fly and visualize the output tensors as physical images.

## 📂 Repository Structure
```text
├── app.py                   # Streamlit dashboard and inference engine
├── gan_architecture.py      # PyTorch DCGAN Generator & Discriminator structures
├── train_gan.py             # Minimax training loop and dataset loader
└── requirements.txt         # Project dependencies
