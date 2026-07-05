# AcoustiGuard — POC Results (Idea Deliberation → Early POC)

## What this is

A working proof of concept for the core AcoustiGuard idea: an autoencoder
trained only on **normal** machine sound can flag **anomalous** machine
sound using reconstruction error, with no labeled fault data.

## About the data

Real industrial datasets like MIMII are hosted on Zenodo, which isn't
reachable from this build environment. To validate the approach quickly,
we generated **synthetic machine sound** that mimics the structure MIMII
data has: a steady harmonic hum for normal operation, and three realistic
fault signatures layered on top for anomalies:

- **Bearing defect** — periodic impact clicks
- **Imbalance** — low-frequency amplitude modulation (wobble)
- **Cavitation / leak** — broadband hiss bursts

This is a stand-in for validating the pipeline end-to-end. The next step
is swapping in real MIMII recordings, which should work with zero changes
to the model or feature pipeline.

## Pipeline

1. `generate_data.py` — synthesizes 120 normal + 60 anomalous audio clips
2. `train_and_eval.py` — converts audio to log-mel spectrograms, trains a
   lightweight autoencoder on normal-only data, evaluates on held-out
   normal + anomalous clips
3. `export_onnx.py` — exports the trained model to ONNX for edge deployment

## Results

- **Held-out normal false positive rate: 0/24** — the model doesn't false-alarm
  on healthy machine sound it hasn't seen before
- **Anomaly detection rate: 46/60 (76.7%)** across three distinct fault types
- **Overall accuracy: 83.3%**
- **Model size: 3.4 KB** as ONNX — trivially deployable on microcontroller-class
  edge hardware

See `anomaly_score_distribution.png` for the separation between normal and
anomalous reconstruction error, and `spectrogram_comparison.png` for what
the model actually sees.

## Honest limitations (for the presentation Q&A)

- Trained and tested on synthetic data, not real MIMII recordings yet
- Cavitation-type anomalies had the lowest detection rate among the three
  fault types — subtler spectral signature, likely needs a slightly deeper
  encoder or more training epochs on real data
- INT8 quantization step hit a tooling issue in this sandbox and needs to
  be re-run in a normal dev environment before the Stage 2 POC
