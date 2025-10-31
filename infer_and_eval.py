# infer_and_eval.py
# Denoise the full lowSNR movie using the trained LightCAD checkpoint.
# If highSNR is aligned, compute PSNR/SSIM over the processed crop.

import os, numpy as np, tifffile as tiff, torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from lightcad_model import LightCAD

# CUDA toggles
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---- PATHS + checkpoint ----
LOW_SNR_PATH  = r"C:\Users\ayush\OneDrive\Desktop\Ayush\Research Paper\Denoising in Calcium Imaging\LightCAD\Data\Zebrafish\01_ZebrafishOT_GCaMP6s_492x492x6955_lowSNR.tif"
HIGH_SNR_PATH = r"C:\Users\ayush\OneDrive\Desktop\Ayush\Research Paper\Denoising in Calcium Imaging\LightCAD\Data\Zebrafish\01_ZebrafishOT_GCaMP6s_492x492x6955_highSNR.tif"  # optional
CKPT          = r"checkpoints\lightcad_e3.pt"   # <- set to your best checkpoint from training
OUT_TIF       = r"denoised_zebrafish_lightcad.tif"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# 4 GB VRAM friendly windows
TWIN, HWIN, WWIN = 96, 160, 160
TSTEP            = 80
BASE_CH          = 12

def normalize_minmax(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x

def load_tiff_as_THW(path):
    arr = tiff.imread(path)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[-1] > 10 and arr.shape[0] <= 6:
        arr = np.moveaxis(arr, -1, 0)
    assert arr.ndim == 3, f"Expected 3D stack, got shape {arr.shape}"
    return arr

@torch.no_grad()
def main():
    # Load low-SNR
    low = load_tiff_as_THW(LOW_SNR_PATH)
    print(f"[infer] Loaded lowSNR shape: T={low.shape[0]}, H={low.shape[1]}, W={low.shape[2]}")
    low_n = normalize_minmax(low).astype(np.float32)
    T, H, W = low_n.shape

    # Model
    model = LightCAD(base=BASE_CH).to(DEVICE).eval()
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state["model"])

    out = np.zeros_like(low_n, dtype=np.float32)
    cnt = np.zeros_like(low_n, dtype=np.float32)

    # center crop region to fit VRAM
    h0 = max(0, (H - HWIN)//2); h1 = h0 + HWIN
    w0 = max(0, (W - WWIN)//2); w1 = w0 + WWIN

    # temporal sliding (with tail coverage)
    t_starts = list(range(0, max(1, T - TWIN + 1), TSTEP))
    if t_starts[-1] != T - TWIN:
        t_starts.append(max(0, T - TWIN))

    for t0 in tqdm(t_starts, desc="Denoising"):
        t1 = min(t0 + TWIN, T)
        tile = low_n[t0:t1, h0:h1, w0:w1]
        if tile.shape[0] < TWIN:
            pad_t = TWIN - tile.shape[0]
            tile = np.pad(tile, ((0, pad_t), (0, 0), (0, 0)), mode="edge")

        inp = torch.from_numpy(tile[None, None]).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(inp).cpu().numpy()[0, 0]
        pred = pred[:(t1 - t0)]

        out[t0:t1, h0:h1, w0:w1] += pred
        cnt[t0:t1, h0:h1, w0:w1] += 1.0

    den = out / np.maximum(cnt, 1e-8)

    # save as 16-bit
    den16 = np.clip(den * 65535.0, 0, 65535).astype(np.uint16)
    tiff.imwrite(OUT_TIF, den16, imagej=True)
    print("Saved denoised movie:", OUT_TIF)

    # Optional eval vs high-SNR (aligned)
    # ---------- Evaluation vs High-SNR (aligned) ----------
    try:
        hi = load_tiff_as_THW(HIGH_SNR_PATH).astype(np.float32)
        print(f"[infer] Loaded highSNR shape: T={hi.shape[0]}, H={hi.shape[1]}, W={hi.shape[2]}")
        hi_n = normalize_minmax(hi).astype(np.float32)

        # ensure we compare same spatial window used for inference
        hi_crop  = hi_n[:, h0:h1, w0:w1]
        den_crop = den[:,  h0:h1, w0:w1]
        low_crop = low_n[:, h0:h1, w0:w1]

        # if shapes still differ (rare), align to the min common crop
        Tm = min(hi_crop.shape[0], den_crop.shape[0], low_crop.shape[0])
        Hm = min(hi_crop.shape[1], den_crop.shape[1], low_crop.shape[1])
        Wm = min(hi_crop.shape[2], den_crop.shape[2], low_crop.shape[2])
        hi_crop  = hi_crop[:Tm, :Hm, :Wm]
        den_crop = den_crop[:Tm, :Hm, :Wm]
        low_crop = low_crop[:Tm, :Hm, :Wm]

        # PSNR / SSIM
        ps_list, ss_list = [], []
        for k in range(Tm):
            ps_list.append(psnr(hi_crop[k], den_crop[k], data_range=1.0))
            ss_list.append(ssim(hi_crop[k], den_crop[k], data_range=1.0))
        print(f"Avg PSNR (crop): {np.mean(ps_list):.3f} dB | Avg SSIM (crop): {np.mean(ss_list):.4f}")

        # Pearson r
        def pearson_r(a, b):
            a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32)
            a -= a.mean(); b -= b.mean()
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float(np.dot(a, b) / denom)

        r_den = [pearson_r(den_crop[k], hi_crop[k]) for k in range(Tm)]
        print(f"Pearson r (denoised vs highSNR): mean={np.mean(r_den):.4f} ± {np.std(r_den):.4f} "
              f"(min={np.min(r_den):.4f}, max={np.max(r_den):.4f})")

        r_low = [pearson_r(low_crop[k], hi_crop[k]) for k in range(Tm)]
        print(f"Pearson r (lowSNR vs highSNR):  mean={np.mean(r_low):.4f} ± {np.std(r_low):.4f} "
              f"(min={np.min(r_low):.4f}, max={np.max(r_low):.4f})")

    except Exception as e:
        print("Evaluation skipped (no highSNR match or size mismatch):", repr(e))

if __name__ == "__main__":
    main()
