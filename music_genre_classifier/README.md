# 🎵 AI Music Genre Classifier

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Librosa](https://img.shields.io/badge/Audio_Processing-Librosa-9B72CB)
![Scikit-Learn](https://img.shields.io/badge/Deep_Learning-MLP-F7931E)

An Audio Signal Processing application that automatically classifies raw music files into genres using a Multi-Layer Perceptron (MLP) Deep Neural Network. 

By leveraging the `librosa` library, the pipeline converts raw audio waveforms into a time-frequency domain representation, extracting Mel-Frequency Cepstral Coefficients (MFCCs) to capture the perceptual rhythm and harmonics of the sound.

## 🚀 Key Features
*   **Audio Signal Processing:** Ingests raw `.wav` and `.mp3` files, resampling them to a standard 22,050 Hz via `librosa.load`.
*   **Feature Extraction:** Mathematically isolates 20 MFCCs per audio slice, calculating the mean matrix to form a scalable 1D feature vector for tabular ML modeling.
*   **Deep Learning Classification:** Utilizes `sklearn.neural_network.MLPClassifier` trained on extracted MFCC arrays to map audio signatures to categorical genres.
*   **Interactive Player:** Features an integrated Streamlit media player and confidence-scoring dashboard.

## 📂 Repository Structure
```text
├── app.py                 # Streamlit UI, audio upload logic, and Librosa extraction
├── train_model.py         # Neural network training loop and synthetic MFCC generation
├── requirements.txt       # Project dependencies
└── model/                     
    ├── music_mlp_model.pkl    # Compiled MLP Neural Network
    └── audio_scaler.pkl       # StandardScaler for feature normalization
