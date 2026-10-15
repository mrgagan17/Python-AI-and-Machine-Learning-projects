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

📊 Real Estate Market Methodology
The dataset generation and model logic reflect actual real estate valuation mechanics:$$\text{Price} = (\text{Total SqFt} \times \text{Base Locality Rate}) + (\text{BHK} \times 200,000) + (\text{Bathrooms} \times 150,000) + \epsilon$$Where:$\text{Base Locality Rate}$: Baseline land/construction cost per sq. ft. specific to each zone (e.g., ₹18,500/sq. ft. in Koramangala vs. ₹3,200/sq. ft. in Doddaballapur).$\text{BHK}$ & $\text{Bathrooms}$: Allowances for interior layout complexity.$\epsilon$: Normal distribution variance representing individual property variations (floor rise, facing, amenities).To ensure realistic bounds, the training generator enforces natural property sizing rules (e.g., a 2 BHK is restricted to standard 900–1,400 sq. ft. boundaries, preventing synthetic skewing).⚙️ Installation & Local Setup1. PrerequisitesPython 3.9 or higher installed.2. Clone the RepositoryBashgit clone [https://github.com/your-username/house-price-predictor.git](https://github.com/your-username/house-price-predictor.git)
cd house-price-predictor
3. Create a Virtual Environment (Recommended)Bash# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
4. Install DependenciesBashpip install -r requirements.txt
🚀 How to Run the ApplicationStep 1: Train the Model & Generate ArtifactsRun the training script to generate the synthetic housing dataset, apply One-Hot Encoding, train the Linear Regression model, and export model.pkl and columns.json:Bashpython model/train_model.py
Expected Terminal Output:Plaintext✅ Retrained Model R² Score: 0.9850
✅ Model artifacts updated successfully!
Step 2: Launch the Web DashboardInvoke Streamlit using Python's module launcher:Bashpython -m streamlit run app.py
Your browser will open automatically at: http://localhost:8501🤖 Model Training & Technical DetailsAlgorithm: Ordinary Least Squares (OLS) Linear Regression (sklearn.linear_model.LinearRegression).Feature Pipeline:total_sqft $\rightarrow$ Continuous Numericalbhk $\rightarrow$ Discrete Numericalbath $\rightarrow$ Discrete Numericallocation $\rightarrow$ Categorical One-Hot Encoded $N-1$ binary featuresTarget Variable: price (Total property value in INR)Validation Metric: $R^2$ Score (typically $> 0.98$ on train-test split).💡 Sample Predictions & Market RangesLocationBuilt-up AreaLayoutEstimated Value RangeEffective RateKoramangala1,200 sq. ft.3 BHK, 2 Bath₹2.20 Cr – ₹2.35 Cr~₹19,000 / sq. ft.Kalyan Nagar1,400 sq. ft.3 BHK, 3 Bath₹1.75 Cr – ₹1.90 Cr~₹13,200 / sq. ft.Kammanahalli1,100 sq. ft.2 BHK, 2 Bath₹1.20 Cr – ₹1.35 Cr~₹11,500 / sq. ft.Electronic City1,000 sq. ft.2 BHK, 2 Bath₹65 L – ₹75 L~₹6,800 / sq. ft.Koramangala (Penthouse)4,900 sq. ft.5 BHK, 5 Bath₹9.10 Cr – ₹9.50 Cr~₹18,900 / sq. ft.☁️ Deployment GuideDeploying to Streamlit Community CloudCommit and push your code to a public GitHub repository.Sign in to Streamlit Community Cloud.Click New app and select your GitHub repository.Set Main file path to app.py.Click Deploy!🛡️ LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
