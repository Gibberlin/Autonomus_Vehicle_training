import argparse, os, time
# Ensure PyTorch cache stays inside workspace
_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "torchinductor")
os.makedirs(_cache, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pilotnet import PilotNet, preprocess

p = argparse.ArgumentParser()
p.add_argument("--data", default="drive_data")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--side-corr", type=float, default=0.05)       # steering correction for left/right cameras
p.add_argument("--straight-weight", type=float, default=0.35)  # sampling weight of near-straight frames
p.add_argument("--val-frac", type=float, default=0.15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-mirror", action="store_true", help="disable horizontal mirror augmentation")
p.add_argument("--bias-weight", type=float, default=0.5, help="penalty on batch mean prediction to prevent directional drift")
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

if not a.no_mirror:
    # Double training dataset with exact horizontal mirroring (x -> flip(x), y -> -y)
    # This guarantees exact 1:1 balance between left and right turns and mean steer == 0.0.
    Xtr_flip = torch.flip(Xtr, dims=[-1])
    ytr_flip = -ytr
    base_tr = torch.cat([base[~is_val], -base[~is_val]], dim=0)
    Xtr = torch.cat([Xtr, Xtr_flip], dim=0)
    ytr = torch.cat([ytr, ytr_flip], dim=0)
    print(f"[Anti-Bias] Applied symmetric horizontal mirroring: {len(Xtr)} balanced training frames (mean steer: {ytr.mean().item():.5f})")
else:
    base_tr = base[~is_val]

wtr = torch.where(base_tr.abs() < 0.02, a.straight_weight, 1.0).to(dev)
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
    pred = predict_val()
    err = (pred - yva).abs()
    turn = yva.abs() > 0.1
    mae = err.mean().item()
    mae_turn = err[turn].mean().item() if turn.any() else float("nan")
    bias = (pred - yva).mean().item()
    return mae, mae_turn, bias

turn = yva.abs() > 0.1
print(f"baseline 'always straight': MAE {yva.abs().mean():.4f}, on turns {yva.abs()[turn].mean():.4f}")

best, history = float("inf"), []
for ep in range(a.epochs):
    idx = torch.multinomial(wtr, nb * a.batch, replacement=True)
    tot = 0.0
    for i in range(nb):
        j = idx[i * a.batch:(i + 1) * a.batch]
        xj = augment(Xtr[j].float() / 255)
        yj = ytr[j]
        pred = model(xj)
        reg = a.bias_weight * (pred.mean() ** 2) if a.bias_weight > 0 else 0.0
        loss = lossf(pred, yj) + reg
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    mae, mae_turn, bias = evaluate()
    score = mae + (0.3 * mae_turn if not np.isnan(mae_turn) else 0.0) + 2.0 * abs(bias)
    history.append([ep + 1, tot / nb, mae, mae_turn, bias])
    flag = ""
    if score < best:
        best = score
        torch.save(model.state_dict(), "steer_model.pt")
        flag = f"  *saved (score={score:.4f})"
    print(f"epoch {ep+1:2d}  train loss {tot/nb:.4f}  val MAE {mae:.4f}  val MAE on turns {mae_turn:.4f}  bias {bias:+.4f}{flag}")

# ---- Diagnostics on the FINAL-epoch model ----
torch.save(model.state_dict(), "steer_last.pt")
pd.DataFrame(history, columns=["epoch", "train_loss", "val_mae", "val_mae_turns", "val_bias"]).to_csv(
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
pv_best = predict_val().cpu().numpy()
bias_best = (pv_best - t).mean()
print(f"\n[Best Checkpoint steer_model.pt Evaluation]")
print(f"  Overall Validation MAE: {np.abs(pv_best - t).mean():.4f}")
print(f"  Validation Mean Bias (Pred - Truth): {bias_best:+.4f}")
print(f"  Right turns predicted (>0.02): {(pv_best > 0.02).sum()}")
print(f"  Left turns predicted (<-0.02): {(pv_best < -0.02).sum()}")
print(f"  Straight predicted (-0.02..0.02): {((pv_best >= -0.02) & (pv_best <= 0.02)).sum()}")
pd.DataFrame({"truth": t, "pred": pv_best}).to_csv("results/val_preds_best.csv", index=False)
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