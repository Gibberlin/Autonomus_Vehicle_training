import argparse, time
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from pilotnet import PilotNet, preprocess

p = argparse.ArgumentParser()
p.add_argument("--data", default="drive_data")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--side-corr", type=float, default=0.05)  # steering correction for left/right cameras
p.add_argument("--val-frac", type=float, default=0.15)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(f"{a.data}/labels.csv")

# Time-ordered split: the last 15% of the drive is validation, with a gap so near-identical
# neighbouring frames don't leak between train and val.
centers = df[df.cam == "center"].sort_values("frame").reset_index(drop=True)
cut = len(centers) - int(len(centers) * a.val_frac)
val_frames = set(centers.frame[cut:])
gap_frames = set(centers.frame[max(0, cut - 20):cut])

# Left camera sits left of centre, so it should steer right to recover (+), and vice versa.
corr = {"center": 0.0, "left": a.side_corr, "right": -a.side_corr}

X, y, base, is_val = [], [], [], []
for r in df.itertuples():
    if r.frame in gap_frames:
        continue
    in_val = r.frame in val_frames
    if in_val and r.cam != "center":
        continue                      # validate on the centre camera only
    img = cv2.imread(f"{a.data}/img/{r.file}")
    if img is None:
        continue
    X.append(preprocess(img))
    y.append(r.steer + corr[r.cam])
    base.append(r.steer)
    is_val.append(in_val)

X = torch.from_numpy(np.stack(X)); y = torch.tensor(y, dtype=torch.float32)
base = torch.tensor(base); is_val = torch.tensor(is_val)
Xtr, ytr = X[~is_val].to(dev), y[~is_val].to(dev)
Xva, yva = X[is_val].to(dev), y[is_val].to(dev)
wtr = torch.where(base[~is_val].abs() < 0.02, 0.35, 1.0).to(dev)   # downweight straight driving
print(f"train {len(Xtr)}  val {len(Xva)}  device {dev}")

model = PilotNet().to(dev)
nb = len(Xtr) // a.batch
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * nb)
lossf = nn.SmoothL1Loss(beta=0.05)

def augment(x):
    n = x.shape[0]
    c = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)   # contrast
    b = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)   # brightness
    return (((x - 0.5) * c + 0.5) * b).clamp(0, 1)

@torch.no_grad()
def evaluate():
    model.eval()
    pred = torch.cat([model(Xva[i:i + 256].float() / 255) for i in range(0, len(Xva), 256)])
    model.train()
    err = (pred - yva).abs()
    turn = yva.abs() > 0.1
    return err.mean().item(), err[turn].mean().item() if turn.any() else float("nan")

turn = yva.abs() > 0.1
print(f"baseline 'always straight': MAE {yva.abs().mean():.4f}, on turns {yva.abs()[turn].mean():.4f}")

best = 9.0
for ep in range(a.epochs):
    idx = torch.multinomial(wtr, nb * a.batch, replacement=True)
    tot = 0.0
    for i in range(nb):
        j = idx[i * a.batch:(i + 1) * a.batch]
        xb = augment(Xtr[j].float() / 255)
        loss = lossf(model(xb), ytr[j])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    mae, mae_turn = evaluate()
    flag = ""
    if mae < best:
        best = mae; torch.save(model.state_dict(), "steer_model.pt"); flag = "  *saved"
    print(f"epoch {ep+1:2d}  train loss {tot/nb:.4f}  val MAE {mae:.4f}  val MAE on turns {mae_turn:.4f}{flag}")

# Latency of the final model (this feeds your 33 ms budget) and ONNX export for the edge step
model.load_state_dict(torch.load("steer_model.pt", map_location=dev)); model.eval()
dummy = torch.zeros(1, 3, 66, 200, device=dev)
with torch.no_grad():
    for _ in range(20): model(dummy)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): model(dummy)
    if dev == "cuda": torch.cuda.synchronize()
print(f"steering model: {(time.perf_counter() - t0) / 200 * 1000:.2f} ms/frame on {dev}")
try:
    torch.onnx.export(model.cpu(), torch.zeros(1, 3, 66, 200), "steer_model.onnx",
                      input_names=["image"], output_names=["steer"], opset_version=17)
    print("exported steer_model.onnx")
except Exception as e:
    print("ONNX export skipped:", e, "(try: pip install onnx onnxscript)")