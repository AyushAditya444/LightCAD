# check_io_and_gpu.py
import time, numpy as np, tifffile as tiff, torch

LOW_SNR_PATH  = r"C:\Users\ayush\OneDrive\Desktop\Ayush\Research Paper\Denoising in Calcium Imaging\LightCAD\Data\Zebrafish\01_ZebrafishOT_GCaMP6s_492x492x6955_lowSNR.tif"
HIGH_SNR_PATH = r"C:\Users\ayush\OneDrive\Desktop\Ayush\Research Paper\Denoising in Calcium Imaging\LightCAD\Data\Zebrafish\01_ZebrafishOT_GCaMP6s_492x492x6955_highSNR.tif"

def load_tiff_as_THW(path):
    t0 = time.time()
    print(f"[check] Loading TIFF: {path}")
    arr = tiff.imread(path)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[-1] > 10 and arr.shape[0] <= 6:
        arr = np.moveaxis(arr, -1, 0)
    assert arr.ndim == 3, f"Expected 3D stack, got shape {arr.shape}"
    dt = time.time() - t0
    print(f"[check] Loaded THW={arr.shape} in {dt:.2f}s; dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    return arr

def main():
    print("[check] Torch:", torch.__version__)
    print("[check] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[check] GPU:", torch.cuda.get_device_name(0))

    low = load_tiff_as_THW(LOW_SNR_PATH)
    T,H,W = low.shape
    t0 = 100
    TSLICE, HS, WS = 32, 160, 160
    h0 = max(0, (H-HS)//2); w0 = max(0, (W-WS)//2)
    tile = low[t0:t0+TSLICE, h0:h0+HS, w0:w0+WS].astype(np.float32)
    print(f"[check] One tile ready: {tile.shape}, mean={tile.mean():.4f}, std={tile.std():.4f}")
    print("[check] All good. Proceed to train_lightcad.py")

if __name__ == "__main__":
    main()
