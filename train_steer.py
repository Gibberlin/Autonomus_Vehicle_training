import argparse, os, time
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pilotnet import PilotNet, preprocess

p = argparse.ArgumentParser()
p.add_argument("--data", default="drive_data")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--side-corr", type=float, default=0.05)       # steering correction for left/right cameras
p.add_argument("--straight-weight", type=float, default=0.35)  # sampling weight of near-straight frames
p.add_argument("--val-frac", type=float, default=0.15)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs("results", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(f"{a.data}/labels.csv")

# Time-ordered split: last part of the drive is validation, with a gap to avoid leakage.
centers = df[df.cam == "center"].sort_values("frame").reset_index(drop=True)
cut = len(centers) - int(len(centers) * a.val_frac)
val_frames = set(centers.frame[cut:])
gap_frames = set(centers.frame[max(0, cut - 20):cut])
corr = {"center": 0.0, "left": a.side_corr, "right": -a.side_corr}

X, y, base, is_val = [], [], [], []
for r in df.itertuples():
    if r.frame in gap_frames:
        continue
    in_val = r.frame in val_frames
    if in_val and r.cam != "center":
        continue
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
wtr = torch.where(base[~is_val].abs() < 0.02, a.straight_weight, 1.0).to(dev)
print(f"train {len(Xtr)}  val {len(Xva)}  device {dev}")

model = PilotNet().to(dev)
nb = len(Xtr) // a.batch
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * nb)
lossf = nn.SmoothL1Loss(beta=0.05)

def augment(x):
    n = x.shape[0]
    c = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    b = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)
    return (((x - 0.5) * c + 0.5) * b).clamp(0, 1)

@torch.no_grad()
def predict_val():
    model.eval()
    out = torch.cat([model(Xva[i:i + 256].float() / 255) for i in range(0, len(Xva), 256)])
    model.train()
    return out

def evaluate():
    err = (predict_val() - yva).abs()
    turn = yva.abs() > 0.1
    return err.mean().item(), err[turn].mean().item() if turn.any() else float("nan")

turn = yva.abs() > 0.1
print(f"baseline 'always straight': MAE {yva.abs().mean():.4f}, on turns {yva.abs()[turn].mean():.4f}")

best, history = 9.0, []
for ep in range(a.epochs):
    idx = torch.multinomial(wtr, nb * a.batch, replacement=True)
    tot = 0.0
    for i in range(nb):
        j = idx[i * a.batch:(i + 1) * a.batch]
        loss = lossf(model(augment(Xtr[j].float() / 255)), ytr[j])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    mae, mae_turn = evaluate()
    history.append([ep + 1, tot / nb, mae, mae_turn])
    flag = ""
    if mae < best:
        best = mae; torch.save(model.state_dict(), "steer_model.pt"); flag = "  *saved"
    print(f"epoch {ep+1:2d}  train loss {tot/nb:.4f}  val MAE {mae:.4f}  val MAE on turns {mae_turn:.4f}{flag}")

# ---- Diagnostics on the FINAL-epoch model ----
torch.save(model.state_dict(), "steer_last.pt")
pd.DataFrame(history, columns=["epoch", "train_loss", "val_mae", "val_mae_turns"]).to_csv(
    "results/steer_training_curve.csv", index=False)
pv = predict_val().cpu().numpy(); t = yva.cpu().numpy(); ya = np.abs(t)
pd.DataFrame({"truth": t, "pred": pv}).to_csv("results/val_preds_last.csv", index=False)

print("\nFinal-epoch model, error by steering size (model MAE vs always-straight):")
for name, m in {"straight (<0.02)": ya < 0.02, "mild (0.02-0.1)": (ya >= 0.02) & (ya <= 0.1),
                "turns (>0.1)": ya > 0.1}.items():
    if m.any():
        print(f"  {name:17s} n={int(m.sum()):4d}  model {np.abs(pv[m] - t[m]).mean():.4f}  baseline {ya[m].mean():.4f}")
m = ya > 0.1
if m.any():
    print(f"  turns where predicted direction matches truth: {np.mean(np.sign(pv[m]) == np.sign(t[m])):.0%}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
ax[0].plot(t, label="autopilot (truth)"); ax[0].plot(pv, label="model", alpha=0.7)
ax[0].set_xlabel("validation sample (time order)"); ax[0].set_ylabel("steer"); ax[0].legend()
ax[1].scatter(t, pv, s=6, alpha=0.5); ax[1].plot([-0.7, 0.7], [-0.7, 0.7], "k--", lw=1)
ax[1].set_xlabel("truth"); ax[1].set_ylabel("model")
fig.savefig("results/val_pred.png", dpi=120, bbox_inches="tight")
print("saved results/val_pred.png, val_preds_last.csv, steer_training_curve.csv")

# ---- Best checkpoint: latency + ONNX export ----
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
    print("ONNX export skipped:", e)