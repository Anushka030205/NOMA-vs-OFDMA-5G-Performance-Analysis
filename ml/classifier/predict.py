import joblib
import numpy as np

# Load trained model
model = joblib.load("noma_ofdm_model.pkl")

# Example input: [SNR, g1, g2, power]
samples = np.array([
    [3, 0.09, 2.2, 0.87],   # low SNR → OFDMA
    [15, 0.2, 1.8, 0.8],    # high SNR → NOMA
])

predictions = model.predict(samples)

for i, p in enumerate(predictions):
    print(f"Sample {i+1}: {'NOMA' if p==1 else 'OFDMA'}")