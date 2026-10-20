# NOMA vs OFDMA: Performance Analysis for 5G NR Networks

A comparative performance analysis of **Non-Orthogonal Multiple Access (NOMA)** and **Orthogonal Frequency Division Multiple Access (OFDMA)** for spectral efficiency enhancement in 5G NR wireless networks, with an ML-based adaptive access technique selector.

## Overview

This project evaluates NOMA and OFDMA under Rayleigh fading channels using MATLAB simulations and builds a machine learning pipeline (Random Forest) to adaptively predict the optimal multiple access technique based on channel conditions.

### Key Metrics Compared

- **Spectral Efficiency (Sum Rate)** — bits/s/Hz
- **Bit Error Rate (BER)**
- **Jain's Fairness Index**
- **Throughput**

### Key Findings

| Metric | NOMA | OFDMA |
|--------|------|-------|
| Spectral Efficiency | Higher (power-domain multiplexing) | Lower (orthogonal subcarrier split) |
| BER | Higher (SIC error propagation) | Lower (no inter-user interference) |
| Fairness | Lower | Higher (equal resource allocation) |
| Throughput | Higher at medium-high SNR | More stable across SNR range |

## Project Structure

```
.
├── simulation/                 # MATLAB simulation code
│   ├── noma_ofdm_main.m        # Main simulation: NOMA vs OFDMA (BER, SE, fairness, throughput)
│   ├── generate_dataset.m      # Generate classification dataset (NOMA-better vs OFDMA-better)
│   └── dataset_nomaofdm.m      # Generate regression dataset (rate difference)
│
├── ml/                         # Machine learning models
│   ├── classifier/
│   │   ├── train_model.py      # Random Forest classifier — predicts NOMA vs OFDMA
│   │   └── predict.py          # Inference script with sample inputs
│   └── regressor/
│       └── train_regressor.py  # Random Forest regressor — predicts rate difference
│
├── data/                       # Generated datasets
│   ├── dataset.csv             # Classification dataset (SNR, g1, g2, power_alloc, label)
│   └── noma_ofdm_dataset.csv   # Regression dataset (SNR, users, rate_diff)
│
├── docs/                       # Reports and references
│   ├── report.pdf              # Final case study report
│   ├── presentation.pptx       # Review presentation
│   └── references/             # Reference papers
│
├── results/                    # Directory for simulation output plots
├── requirements.txt            # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- **MATLAB** R2020a or later (for simulations)
- **Python** 3.8+ (for ML models)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/NOMA-vs-OFDMA-5G-Performance-Analysis.git
cd NOMA-vs-OFDMA-5G-Performance-Analysis

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Simulation

Open MATLAB and run:

```matlab
% Full simulation with plots
run('simulation/noma_ofdm_main.m')

% Generate fresh datasets
run('simulation/generate_dataset.m')       % classification dataset
run('simulation/dataset_nomaofdm.m')       % regression dataset
```

### Training the ML Model

```bash
# Train the Random Forest classifier
cd ml/classifier
python train_model.py

# Run predictions on sample inputs
python predict.py
```

### Training the Regressor

```bash
cd ml/regressor
python train_regressor.py
```

## System Model

The simulation models a two-user downlink scenario:

- **NOMA**: Superposition coding with power allocation coefficients (alpha_1 = 0.75 for far user, alpha_2 = 0.25 for near user). SIC decoding at the near user.
- **OFDMA**: Equal power split across orthogonal subcarriers, one per user.
- **Channel**: Rayleigh fading with distinct channel gains (weak user: 0.3 scale, strong user: 1.5 scale).
- **SNR Range**: 0 to 20 dB.

## ML Pipeline

The Random Forest classifier takes four features as input:

| Feature | Description |
|---------|-------------|
| SNR (dB) | Signal-to-noise ratio |
| g1 | Weak user channel gain |
| g2 | Strong user channel gain |
| alpha_1 | Far-user power allocation coefficient |

**Output**: Binary label — `1` (NOMA preferred) or `0` (OFDMA preferred).

## Tools & Technologies

- MATLAB (signal simulation, dataset generation)
- Python, scikit-learn (Random Forest classifier/regressor)
- pandas, NumPy, joblib

## Author

**Anushka Vishal Chandergi**

## License

This project is for academic/educational purposes as part of the Wireless and Mobile Communication course.
