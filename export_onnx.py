import torch
import torch.nn as nn
import os

N_MELS = 64
N_FRAMES = 64
IN_DIM = N_MELS * N_FRAMES


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


model = AutoEncoder(IN_DIM)
model.load_state_dict(torch.load("autoencoder.pt"))
model.eval()

dummy = torch.randn(1, IN_DIM)
torch.onnx.export(
    model, dummy, "autoencoder.onnx",
    input_names=["logmel_input"], output_names=["reconstruction"],
    opset_version=17,
)

fp32_size = os.path.getsize("autoencoder.onnx") / 1024
print(f"ONNX model exported: autoencoder.onnx ({fp32_size:.1f} KB)")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("autoencoder.onnx", "autoencoder_int8.onnx", weight_type=QuantType.QInt8)
    int8_size = os.path.getsize("autoencoder_int8.onnx") / 1024
    print(f"INT8 quantized model: autoencoder_int8.onnx ({int8_size:.1f} KB)")
    print(f"Size reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
except ImportError:
    print("onnxruntime not installed - skipping INT8 quantization step")
