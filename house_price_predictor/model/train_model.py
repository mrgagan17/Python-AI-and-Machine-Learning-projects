import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

def build_and_train():
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    np.random.seed(42)
    n_samples = 8000

    # Base price per sq. ft. (INR) across Bengaluru localities
    loc_rates = {
        'Koramangala': 18500,
        'Indiranagar': 19000,
        'Kammanahalli': 11000,
        'Kalyan Nagar': 12500,
        'HRBR Layout': 12000,
        'Banaswadi': 9500,
        'Hennur Road': 10000,
        'MG Road / Lavelle Road': 22000,
        'Malleshwaram': 14000,
        'Rajajinagar': 13000,
        'Whitefield': 12000,
        'Outer Ring Road (ORR)': 13000,
        'Sarjapur Road': 10500,
        'KR Puram': 8500,
        'Varthur': 8000,
        'Jayanagar': 14500,
        'JP Nagar': 11000,
        'HSR Layout': 14000,
        'Electronic City': 6500,
        'Kanakapura Road': 8500,
        'Uttarahalli': 7500,
        'Hebbal': 13500,
        'Thanisandra': 9000,
        'Yelahanka': 7800,
        'Devanahalli': 6800,
        'Kengeri': 6200,
        'RR Nagar': 8800,
        'Peenya': 5500,
        'Hoskote': 4800,
        'Attibele': 4200,
        'Chandapura': 4000,
        'Anekal': 3500,
        'Doddaballapur': 3200,
        'Nelamangala': 3000,
        'Bidadi (Rural)': 2800,
        'Magadi Road (Rural)': 2500,
        'Solur (Rural)': 2000
    }

    locations = list(loc_rates.keys())
    
    # Generate realistic data where square footage aligns logically with BHK
    bhk = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.1, 0.35, 0.35, 0.12, 0.05, 0.03])
    
    sqft = []
    for b in bhk:
        # Realistic sqft range based on BHK
        if b == 1:
            s = np.random.randint(500, 850)
        elif b == 2:
            s = np.random.randint(900, 1400)
        elif b == 3:
            s = np.random.randint(1350, 2200)
        elif b == 4:
            s = np.random.randint(2200, 4000)
        elif b == 5:
            s = np.random.randint(3800, 8000)
        else: # 6+ BHK Ultra Villa / Penthouse
            s = np.random.randint(7500, 20000)
        sqft.append(s)

    sqft = np.array(sqft)
    locs = np.random.choice(locations, size=n_samples)
    bath = np.clip(bhk + np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]), 1, 8).astype(int)

    # Realistic Total Price calculation
    price = []
    for s, l, b, bt in zip(sqft, locs, bhk, bath):
        rate = loc_rates[l]
        # Base price calculated strictly on rate/sqft + minor premium for bedrooms/baths + noise
        p = (s * rate) + (b * 200000) + (bt * 150000) + np.random.normal(0, 150000)
        price.append(max(p, 1500000))

    df = pd.DataFrame({
        'total_sqft': sqft,
        'bhk': bhk,
        'bath': bath,
        'location': locs,
        'price': price
    })

    df.to_csv('data/bengaluru_housing.csv', index=False)

    # One-Hot Encoding
    dummies = pd.get_dummies(df['location'], dtype=int)
    df_processed = pd.concat([df.drop('location', axis=1), dummies], axis=1)

    X = df_processed.drop('price', axis=1)
    y = df_processed['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    print(f"✅ Retrained Model R² Score: {score:.4f}")

    joblib.dump(model, 'model/model.pkl')
    with open('model/columns.json', 'w') as f:
        json.dump({'data_columns': [str(col) for col in X.columns]}, f)

    print("✅ Model artifacts updated successfully!")

if __name__ == '__main__':
    build_and_train()