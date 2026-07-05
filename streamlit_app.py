"""
AcoustiGuard — Live Demo Dashboard

Loads the trained autoencoder and lets you feed it any machine-sound clip
(normal or anomalous) to watch the reconstruction-error anomaly score and
alert fire in real time. This is the screen you record for the InnoVent
demo video.

Run order (from the code_files folder):
    pip install streamlit librosa torch soundfile matplotlib --break-system-packages
    python generate_data.py        # creates data/normal, data/anomalous
    python train_and_eval.py       # trains model, saves autoencoder.pt
    streamlit run streamlit_app.py
"""

import torch
torch.classes.__path__ = []
import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SR = 16000
N_MELS = 64
N_FRAMES = 64

st.set_page_config(page_title="AcoustiGuard — Live Demo", layout="wide")


def wav_to_logmel(path):
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=512)
    logmel = librosa.power_to_db(mel, ref=np.max)
    if logmel.shape[1] < N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, N_FRAMES - logmel.shape[1])))
    else:
        logmel = logmel[:, :N_FRAMES]
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)
    return logmel.astype(np.float32)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, in_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@st.cache_resource
def load_model_and_threshold():
    in_dim = N_MELS * N_FRAMES
    model = AutoEncoder(in_dim)
    ckpt_path = os.path.join(BASE_DIR, "autoencoder.pt")
    if not os.path.exists(ckpt_path):
        st.error("autoencoder.pt not found — run `python train_and_eval.py` first.")
        st.stop()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    normal_paths = sorted(glob.glob(os.path.join(BASE_DIR, "data/normal/*.wav")))
    if not normal_paths:
        st.error("No audio found in data/normal — run `python generate_data.py` first.")
        st.stop()
    split = int(len(normal_paths) * 0.8)
    train_paths = normal_paths[:split]
    feats = np.stack([wav_to_logmel(p) for p in train_paths]).reshape(len(train_paths), -1)
    with torch.no_grad():
        out = model(torch.tensor(feats))
        errs = ((out - torch.tensor(feats)) ** 2).mean(dim=1).numpy()
    threshold = errs.mean() + 3 * errs.std()
    return model, threshold


model, threshold = load_model_and_threshold()

st.title("AcoustiGuard — Live Anomaly Detection Demo")
st.caption("Edge-based acoustic anomaly detection · pick a clip, watch the reconstruction error react")

all_normal = sorted(glob.glob(os.path.join(BASE_DIR, "data/normal/*.wav")))
all_anom = sorted(glob.glob(os.path.join(BASE_DIR, "data/anomalous/*.wav")))
options = {f"[normal] {os.path.basename(p)}": p for p in all_normal}
options.update({f"[anomalous] {os.path.basename(p)}": p for p in all_anom})

choice = st.selectbox("Pick a machine-sound clip", list(options.keys()))
path = options[choice]

logmel = wav_to_logmel(path)
x = torch.tensor(logmel.reshape(1, -1))
with torch.no_grad():
    out = model(x)
    err = ((out - x) ** 2).mean().item()

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(logmel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Log-mel spectrogram (what the model sees)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bands")
    st.pyplot(fig)
    st.audio(path)

with col2:
    st.metric("Reconstruction error", f"{err:.5f}")
    st.caption(f"Anomaly threshold: {threshold:.5f}")
    if err > threshold:
        st.error("⚠ ANOMALY DETECTED — sound deviates from the healthy baseline")
    else:
        st.success("✓ Normal — within healthy operating range")
    st.progress(float(min(err / (threshold * 2), 1.0)))
