# sanity_forward.py
import torch
from lightcad_model import LightCAD

print("[sanity] Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("[sanity] GPU:", torch.cuda.get_device_name(0), flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LightCAD(base=12, verbose=True).to(device).eval()

x = torch.randn(1, 1, 16, 64, 64, device=device)  # [B,C,T,H,W]
with torch.inference_mode():
    y = model(x)
print("[sanity] Forward OK | input:", tuple(x.shape), "output:", tuple(y.shape), flush=True)
