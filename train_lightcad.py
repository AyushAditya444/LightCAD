# train_lightcad.py
# Self-supervised training (Noise2Noise style) on your lowSNR TIFF.
# Loads compressed TIFFs with imread, handles axis order, uses AMP, verbose logs,
# quick-mode subset, and one-batch debug exit to guarantee visible output.

import os, random, time, numpy as np, tifffile as tiff, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lightcad_model import LightCAD

# ---- CUDA speed toggles ----
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---- YOUR PATHS ----
LOW_SNR_PATH  = r"C:\Users\ayush\OneDrive\Desktop\Ayush\Research Paper\Denoising in Calcium Imaging\LightCAD\Data\Zebrafish\01_ZebrafishOT_GCaMP6s_492x492x6955_lowSNR.tif"
HIGH_SNR_PATH = r"C:\Users\ayush\OneDrive\Desktop\Ayush\Research Paper\Denoising in Calcium Imaging\LightCAD\Data\Zebrafish\01_ZebrafishOT_GCaMP6s_492x492x6955_highSNR.tif"

SAVE_DIR   = "checkpoints"
LOG_FILE   = "training_log.txt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- 4 GB VRAM friendly hyperparams ----
T, H, W   = 96, 160, 160
STRIDE_T  = 80
BATCH     = 1
EPOCHS    = 3         # start small; bump after confirming
LR        = 5e-5
BASE_CH   = 12

# Debug switches
QUICK_MODE      = True   # use only first 512 frames for quick visible run
DEBUG_ONE_BATCH = False  # if True: save warmup checkpoint after 1 batch and EXIT

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def normalize_minmax(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x

def load_tiff_as_THW(path):
    t0 = time.time()
    log(f"[load] Reading TIFF: {path}")
    arr = tiff.imread(path)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[-1] > 10 and arr.shape[0] <= 6:
        arr = np.moveaxis(arr, -1, 0)
    assert arr.ndim == 3, f"Expected 3D stack, got shape {arr.shape}"
    log(f"[load] Done. THW={arr.shape}, dtype={arr.dtype}, load_time={time.time()-t0:.2f}s")
    return arr

class InterlacedTiles(Dataset):
    """input = [t..t+T], target = [t+1..t+1+T] from low-SNR stack."""
    def __init__(self, tif_path, t=T, h=H, w=W, stride_t=STRIDE_T, train=True, quick_mode=False):
        super().__init__()
        arr = load_tiff_as_THW(tif_path)
        if quick_mode:
            maxT = min(arr.shape[0], 512)
            arr = arr[:maxT]
            log(f"[dataset] QUICK_MODE -> using first {arr.shape[0]} frames")
        self.arr = normalize_minmax(arr).astype(np.float32)

        self.t, self.h, self.w = t, h, w
        self.stride_t = stride_t
        self.train = train

        Ttot, Htot, Wtot = self.arr.shape
        self.h0 = max(0, (Htot - h)//2)
        self.w0 = max(0, (Wtot - w)//2)
        self.starts = list(range(0, Ttot - t - 1, stride_t))
        log(f"[dataset] Ttot={Ttot}, tiles={len(self.starts)}, tile_size={(t,h,w)}, stride_t={stride_t}")

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        a = self.arr[t0      : t0+self.t, self.h0:self.h0+self.h, self.w0:self.w0+self.w]
        b = self.arr[t0 + 1  : t0+1+self.t, self.h0:self.h0+self.h, self.w0:self.w0+self.w]
        if self.train:
            if random.random() < 0.5:
                a = a[:, :, ::-1].copy(); b = b[:, :, ::-1].copy()
            if random.random() < 0.5:
                a = a[:, ::-1, :].copy(); b = b[:, ::-1, :].copy()
        return torch.from_numpy(a[None]), torch.from_numpy(b[None])  # [1,T,H,W]

def loss_fn(pred, target):
    return 0.5*F.l1_loss(pred, target) + 0.5*F.mse_loss(pred, target)

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("[start] training log\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    log(f"[env] Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"[env] GPU: {torch.cuda.get_device_name(0)} | VRAM ~{props.total_memory/1024**3:.1f} GB")

    # Data
    t0 = time.time()
    train_ds = InterlacedTiles(LOW_SNR_PATH, train=True, quick_mode=QUICK_MODE)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    log(f"[data] DataLoader ready in {time.time()-t0:.2f}s, batches/epoch={len(train_dl)}")

    # Model + Optimizer + AMP
    log("[model] Building LightCAD ...")
    model = LightCAD(base=BASE_CH, verbose=True).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    log("[model] Ready.")

    best_path, best_loss = None, float("inf")
    saved_warmup = False

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        log(f"[epoch] >>> Start Epoch {epoch}/{EPOCHS}")
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", mininterval=1.0)
        for ib, (a, b) in enumerate(pbar):
            a = a.to(DEVICE, non_blocking=True).float()
            b = b.to(DEVICE, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                y = model(a)
                loss = loss_fn(y, b)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Early checkpoint; optional early exit for debug
            if not saved_warmup and ib == 0:
                warm = os.path.join(SAVE_DIR, "lightcad_warmup.pt")
                torch.save({"model": model.state_dict()}, warm)
                log(f"[ckpt] Saved warmup checkpoint: {warm}")
                saved_warmup = True
                if DEBUG_ONE_BATCH:
                    log("[debug] DEBUG_ONE_BATCH=True -> exiting right after warmup checkpoint.")
                    return

        avg = running / max(1, len(train_dl))
        ckpt = os.path.join(SAVE_DIR, f"lightcad_e{epoch}.pt")
        torch.save({"model": model.state_dict()}, ckpt)
        if avg < best_loss:
            best_loss, best_path = avg, ckpt
        log(f"[epoch] {epoch}: avg_loss={avg:.4f} | best={os.path.basename(best_path)} ({best_loss:.4f})")

    log(f"[done] Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
