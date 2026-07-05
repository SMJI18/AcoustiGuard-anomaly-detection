"""
AcoustiGuard POC: unsupervised acoustic anomaly detection.

Pipeline: wav -> log-mel spectrogram -> lightweight autoencoder
trained ONLY on normal machine sound -> reconstruction error as
anomaly score.
"""
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SR = 16000
N_MELS = 64
N_FRAMES = 64  # fixed width, ~2 sec of audio at these hop settings

device = torch.device("cpu")


def wav_to_logmel(path):
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=512)
    logmel = librosa.power_to_db(mel, ref=np.max)
    # pad/crop to fixed width
    if logmel.shape[1] < N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, N_FRAMES - logmel.shape[1])))
    else:
        logmel = logmel[:, :N_FRAMES]
    # normalize to 0-1
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
        z = self.encoder(x)
        return self.decoder(z)


def load_set(paths):
    feats = [wav_to_logmel(p) for p in paths]
    feats = np.stack(feats)
    return feats.reshape(feats.shape[0], -1)


normal_paths = sorted(glob.glob("data/normal/*.wav"))
anom_paths = sorted(glob.glob("data/anomalous/*.wav"))

# 80/20 split on normal data: train on the 80%, treat held-out 20% as
# "normal test" to check we don't false-alarm on healthy machines
split = int(len(normal_paths) * 0.8)
train_paths = normal_paths[:split]
normal_test_paths = normal_paths[split:]

X_train = load_set(train_paths)
X_normal_test = load_set(normal_test_paths)
X_anom_test = load_set(anom_paths)

in_dim = X_train.shape[1]
print(f"Feature dim: {in_dim}, train samples: {len(X_train)}")

model = AutoEncoder(in_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train)
EPOCHS = 60
for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    out = model(X_train_t)
    loss = loss_fn(out, X_train_t)
    loss.backward()
    opt.step()
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1}/{EPOCHS} - train loss: {loss.item():.5f}")

model.eval()


def recon_errors(X):
    with torch.no_grad():
        X_t = torch.tensor(X)
        out = model(X_t)
        errs = ((out - X_t) ** 2).mean(dim=1).numpy()
    return errs


train_errs = recon_errors(X_train)
normal_test_errs = recon_errors(X_normal_test)
anom_errs = recon_errors(X_anom_test)

threshold = train_errs.mean() + 3 * train_errs.std()

print("\n--- Results ---")
print(f"Train (normal) recon error:      mean={train_errs.mean():.5f}  std={train_errs.std():.5f}")
print(f"Held-out normal recon error:     mean={normal_test_errs.mean():.5f}  std={normal_test_errs.std():.5f}")
print(f"Anomalous recon error:           mean={anom_errs.mean():.5f}  std={anom_errs.std():.5f}")
print(f"Anomaly threshold (train mean + 3*std): {threshold:.5f}")

normal_test_flagged = (normal_test_errs > threshold).sum()
anom_flagged = (anom_errs > threshold).sum()
print(f"\nFalse positives on held-out normal: {normal_test_flagged}/{len(normal_test_errs)}")
print(f"True positives on anomalous:        {anom_flagged}/{len(anom_errs)}")

accuracy = (
    (normal_test_errs <= threshold).sum() + (anom_errs > threshold).sum()
) / (len(normal_test_errs) + len(anom_errs))
print(f"Overall detection accuracy: {accuracy*100:.1f}%")

# ---- Plot ----
plt.figure(figsize=(9, 5))
plt.hist(normal_test_errs, bins=15, alpha=0.6, label="Held-out normal", color="#028090")
plt.hist(anom_errs, bins=15, alpha=0.6, label="Anomalous", color="#E8A33D")
plt.axvline(threshold, color="red", linestyle="--", label="Anomaly threshold")
plt.xlabel("Reconstruction error")
plt.ylabel("Count")
plt.title("AcoustiGuard: Reconstruction Error Distribution\n(Normal vs Anomalous Machine Sound)")
plt.legend()
plt.tight_layout()
plt.savefig("anomaly_score_distribution.png", dpi=150)
print("\nSaved plot: anomaly_score_distribution.png")

# Save model + a sample spectrogram plot for the presentation
torch.save(model.state_dict(), "autoencoder.pt")

sample_normal = wav_to_logmel(normal_paths[0])
sample_anom = wav_to_logmel(anom_paths[0])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(sample_normal, aspect="auto", origin="lower", cmap="magma")
axes[0].set_title("Normal Machine Sound (log-mel spectrogram)")
axes[1].imshow(sample_anom, aspect="auto", origin="lower", cmap="magma")
axes[1].set_title("Anomalous Machine Sound (log-mel spectrogram)")
plt.tight_layout()
plt.savefig("spectrogram_comparison.png", dpi=150)
print("Saved plot: spectrogram_comparison.png")
