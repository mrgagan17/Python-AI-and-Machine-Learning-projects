# 🏠 Real-World Metropolitan House Price Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed to estimate residential real estate market valuations across **Bengaluru Urban, Bengaluru Rural, and the Outer Metropolitan Region (BMRDA limits)** in **Indian Rupees (₹)**.

This application uses a multivariate **Linear Regression** model coupled with categorical **One-Hot Encoding** to evaluate properties ranging from 500 sq. ft. studio apartments to 20,000 sq. ft. luxury estates, penthouses, and suburban villas.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Real Estate Market Methodology](#-real-estate-market-methodology)
- [Installation & Local Setup](#-installation--local-setup)
- [How to Run the Application](#-how-to-run-the-application)
- [Model Training & Technical Details](#-model-training--technical-details)
- [Sample Predictions & Market Ranges](#-sample-predictions--market-ranges)
- [Deployment Guide](#-deployment-guide)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Key Features

1. **Exhaustive Geographical Coverage (35+ Zones):**
   - **Central & Ultra-Prime:** Koramangala, Indiranagar, MG Road/Lavelle Road, Malleshwaram, Sadashivanagar, Rajajinagar.
   - **North-East Commercial & Residential:** Kammanahalli, Kalyan Nagar, HRBR Layout, Banaswadi, Hennur Road.
   - **IT Corridors & Growth Belts:** Whitefield, Outer Ring Road (ORR), Sarjapur Road, KR Puram, Varthur, Electronic City.
   - **South & North Corridors:** Jayanagar, JP Nagar, HSR Layout, Hebbal, Thanisandra, Yelahanka, Devanahalli.
   - **Suburban, Peri-Urban & Rural Belts:** Hoskote, Attibele, Chandapura, Anekal, Doddaballapur, Nelamangala, Bidadi, Solur.

2. **Extended Property Scaling:**
   - Supports total built-up area inputs from **500 sq. ft. up to 20,000 sq. ft.**
   - Dynamically scales BHK configurations (1 to 8 BHK) and bathroom counts (1 to 8 Baths).

3. **Intelligent Indian Rupee (INR) Formatting:**
   - Formats valuations into **Crores (Cr)** and **Lakhs (L)** for easy readability.
   - Displays real-time **Effective Price per Square Foot (₹/sq. ft.)**.
   - Generates an estimated valuation confidence range ($\pm 5\%$).

4. **Production-Ready Interface:**
   - Built with Streamlit, featuring reactive input controls, summary cards, and training dataset visualization tabs.
   - Decoupled column schema mapping (`columns.json`) ensures zero column mismatch errors during deployment.

---

## 📁 Project Architecture & Directory Structure

```text
house_price_predictor/
│
├── data/
│   └── bengaluru_housing.csv     # Synthetic dataset based on real metropolitan valuation trends
│
├── model/
│   ├── train_model.py            # Data generation, preprocessing, model training & serialization script
│   ├── model.pkl                 # Trained Scikit-Learn Linear Regression model artifact
│   └── columns.json              # Serialized One-Hot Encoded feature columns schema
│
├── app.py                        # Streamlit web application frontend code
└── requirements.txt              # Required Python dependencies with pinned versions
