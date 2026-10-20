"""Python port of simulation/noma_ofdm_main.m (NOMA vs OFDMA, two-user downlink).

Reproduces the MATLAB simulation (BER, sum rate, Jain fairness, throughput over
SNR = 0..20 dB) without needing a MATLAB licence. Plots are saved to results/.

Usage (from the repository root):
    python simulation/noma_ofdm_main.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ---------------- Parameters ----------------
num_users = 2
N = 10_000
snr_dB = np.arange(0, 21, 2)
snr = 10 ** (snr_dB / 10)
Pt = 1.0
a1, a2 = 0.75, 0.25  # far (weak) user, near (strong) user


def cgauss(size):
    """Unit-variance complex Gaussian samples, CN(0, 1)."""
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)


SE_NOMA = np.zeros(len(snr))
SE_OFDMA = np.zeros(len(snr))
BER_NOMA = np.zeros(len(snr))
BER_OFDMA = np.zeros(len(snr))
fair_noma = np.zeros(len(snr))
fair_ofdma = np.zeros(len(snr))
thr_noma = np.zeros(len(snr))
thr_ofdma = np.zeros(len(snr))

for i, s in enumerate(snr):
    noise_var = Pt / s

    d1 = rng.integers(0, 2, N)
    d2 = rng.integers(0, 2, N)
    s1, s2 = 2 * d1 - 1, 2 * d2 - 1

    h1 = 0.3 * cgauss(N)  # weak user
    h2 = 1.5 * cgauss(N)  # strong user

    n1 = np.sqrt(noise_var) * cgauss(N)
    n2 = np.sqrt(noise_var) * cgauss(N)

    # ---------------- NOMA ----------------
    x = np.sqrt(a1 * Pt) * s1 + np.sqrt(a2 * Pt) * s2
    y1 = h1 * x + n1
    y2 = h2 * x + n2

    d1_hat = (y1 / h1).real > 0

    y2_eq = y2 / h2
    s1_est = 2 * ((y2_eq.real) > 0) - 1
    y2_sic = y2_eq - np.sqrt(a1 * Pt) * s1_est
    d2_hat = y2_sic.real > 0

    BER_NOMA[i] = (np.sum(d1 != d1_hat) + np.sum(d2 != d2_hat)) / (2 * N)

    sinr1 = (a1 * Pt * np.abs(h1) ** 2) / (a2 * Pt * np.abs(h1) ** 2 + noise_var)
    sinr2 = (a2 * Pt * np.abs(h2) ** 2) / noise_var
    R1 = np.mean(np.log2(1 + sinr1))
    R2 = np.mean(np.log2(1 + sinr2))

    SE_NOMA[i] = R1 + R2
    fair_noma[i] = (R1 + R2) ** 2 / (2 * (R1**2 + R2**2))
    thr_noma[i] = SE_NOMA[i] * (1 - (0.05 + 0.05 * rng.random()))

    # ---------------- OFDMA ----------------
    P_user = Pt / num_users
    y1o = h1 * np.sqrt(P_user) * s1 + n1
    y2o = h2 * np.sqrt(P_user) * s2 + n2

    d1_hat_o = (y1o / h1).real > 0
    d2_hat_o = (y2o / h2).real > 0
    BER_OFDMA[i] = (np.sum(d1 != d1_hat_o) + np.sum(d2 != d2_hat_o)) / (2 * N)

    R1o = np.mean(np.log2(1 + P_user * np.abs(h1) ** 2 / noise_var))
    R2o = np.mean(np.log2(1 + P_user * np.abs(h2) ** 2 / noise_var))

    SE_OFDMA[i] = (R1o + R2o) / num_users
    fair_ofdma[i] = (R1o + R2o) ** 2 / (2 * (R1o**2 + R2o**2))
    thr_ofdma[i] = SE_OFDMA[i] * (1 - (0.10 + 0.05 * rng.random()))


# ---------------- Plots ----------------
def make_plot(y_noma, y_ofdma, ylabel, title, fname, log=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plot = ax.semilogy if log else ax.plot
    plot(snr_dB, y_noma, "r-o", lw=2, label="NOMA")
    plot(snr_dB, y_ofdma, "b-s", lw=2, label="OFDMA")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close(fig)


make_plot(SE_NOMA, SE_OFDMA, "bits/s/Hz", "Sum Rate Comparison", "sum_rate.png")
make_plot(BER_NOMA, BER_OFDMA, "Bit Error Rate", "BER Comparison", "ber.png", log=True)
make_plot(fair_noma, fair_ofdma, "Jain Fairness Index", "Fairness Comparison", "fairness.png")
make_plot(thr_noma, thr_ofdma, "Throughput", "Throughput Comparison", "throughput.png")

# Numeric results table
with open(RESULTS_DIR / "simulation_results.csv", "w") as f:
    f.write("snr_dB,SE_NOMA,SE_OFDMA,BER_NOMA,BER_OFDMA,fair_NOMA,fair_OFDMA,thr_NOMA,thr_OFDMA\n")
    for row in zip(snr_dB, SE_NOMA, SE_OFDMA, BER_NOMA, BER_OFDMA, fair_noma, fair_ofdma, thr_noma, thr_ofdma):
        f.write(",".join(f"{v:.4f}" if k else f"{int(v)}" for k, v in enumerate(row)) + "\n")

print("Saved plots and simulation_results.csv to", RESULTS_DIR)
