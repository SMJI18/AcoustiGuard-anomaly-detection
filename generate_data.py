"""
Generates synthetic industrial machine sound for AcoustiGuard POC.

Real MIMII data isn't reachable from this sandbox (no access to Zenodo),
so we simulate the same underlying phenomenon: a machine has a steady
harmonic hum when healthy, and anomalies show up as irregular transients,
extra sidebands, or amplitude modulation riding on top of that hum.
This is enough to validate that the autoencoder approach can separate
normal vs anomalous sound purely from its spectral shape.
"""
import numpy as np
import soundfile as sf
import os

SR = 16000
DURATION = 2.0
N_SAMPLES = int(SR * DURATION)

os.makedirs("data/normal", exist_ok=True)
os.makedirs("data/anomalous", exist_ok=True)

rng = np.random.default_rng(42)


def normal_machine_sound(base_freq=120.0, seed=0):
    r = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # fundamental hum + a few stable harmonics (typical of motor/fan noise)
    sig = np.zeros(N_SAMPLES)
    for i, h in enumerate([1, 2, 3, 4], start=1):
        amp = 1.0 / h
        drift = r.normal(0, 0.5)  # tiny frequency drift, still "normal"
        sig += amp * np.sin(2 * np.pi * (base_freq * h + drift) * t)
    # steady background noise (bearing/air noise)
    sig += r.normal(0, 0.05, N_SAMPLES)
    sig = sig / np.max(np.abs(sig))
    return sig.astype(np.float32)


def anomalous_machine_sound(base_freq=120.0, seed=0, kind="bearing"):
    r = np.random.default_rng(seed)
    sig = normal_machine_sound(base_freq, seed)
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

    if kind == "bearing":
        # sharp periodic impacts (bearing defect click train)
        click_rate = 40  # Hz
        clicks = np.zeros(N_SAMPLES)
        period = int(SR / click_rate)
        for start in range(0, N_SAMPLES, period):
            width = int(SR * 0.002)
            end = min(start + width, N_SAMPLES)
            clicks[start:end] += r.uniform(0.6, 1.0)
        sig = sig * 0.7 + clicks
    elif kind == "imbalance":
        # low-frequency amplitude modulation (rotor imbalance wobble)
        mod = 1 + 0.6 * np.sin(2 * np.pi * 8 * t)
        sig = sig * mod
    elif kind == "cavitation":
        # broadband high-frequency hiss bursts (cavitation/leak)
        hiss = r.normal(0, 0.4, N_SAMPLES)
        envelope = (np.sin(2 * np.pi * 3 * t) > 0.3).astype(float)
        sig = sig + hiss * envelope

    sig = sig / np.max(np.abs(sig))
    return sig.astype(np.float32)


N_NORMAL = 120
N_ANOM_PER_TYPE = 20

for i in range(N_NORMAL):
    freq = 120 + rng.normal(0, 3)
    sig = normal_machine_sound(freq, seed=i)
    sf.write(f"data/normal/normal_{i:03d}.wav", sig, SR)

anomaly_kinds = ["bearing", "imbalance", "cavitation"]
idx = 0
for kind in anomaly_kinds:
    for j in range(N_ANOM_PER_TYPE):
        freq = 120 + rng.normal(0, 3)
        sig = anomalous_machine_sound(freq, seed=1000 + idx, kind=kind)
        sf.write(f"data/anomalous/{kind}_{j:03d}.wav", sig, SR)
        idx += 1

print(f"Generated {N_NORMAL} normal clips and {len(anomaly_kinds) * N_ANOM_PER_TYPE} anomalous clips.")
