import glob, sys, time
import numpy as np
import cv2
from ultralytics import YOLO

device = sys.argv[1] if len(sys.argv) > 1 else "0"
half = len(sys.argv) > 2 and sys.argv[2] == "half"
src = sys.argv[3] if len(sys.argv) > 3 else "clip_frames"
FPS = 20

# Camera geometry (matches record_clip.py: z=2.0 m, FOV 90, 640x480)
FOV, CAM_H, LANE_HALF = 90, 2.0, 1.8
KEEP = [0, 1, 2, 3, 5, 7]        # person, bicycle, car, motorcycle, bus, truck (no traffic lights)
LEAD_CLASSES = {2, 3, 5, 7}      # what counts as "the car ahead"
TICKS = [5, 10, 20, 30]          # metres

kwargs = dict(device=device, verbose=False, conf=0.25, classes=KEEP)
if half:
    kwargs["quantize"] = 16

files = [f for f in sorted(glob.glob(f"{src}/*.png")) if cv2.imread(f) is not None]
print(f"{len(files)} frames")
model = YOLO("yolov8n.pt")
for f in files[:10]:
    model(f, **kwargs)

h, w = cv2.imread(files[0]).shape[:2]
cx, cy = w / 2, h / 2
focal = w / (2 * np.tan(np.radians(FOV / 2)))

def ground(Z, X):
    """Project a road point (Z metres ahead, X metres right) to pixel coords."""
    return int(cx + focal * X / Z), int(cy + focal * CAM_H / Z)

def dist_color(Z):
    return (0, 0, 255) if Z < 8 else (0, 255, 255) if Z < 15 else (0, 255, 0)

out_name = f"detections_{'cpu' if device == 'cpu' else 'gpu'}{'_fp16' if half else ''}.mp4"
vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

for f in files:
    t0 = time.perf_counter()
    r = model(f, **kwargs)[0]
    ms = (time.perf_counter() - t0) * 1000
    frame = r.plot()

    # Find the lead vehicle: nearest box inside our lane corridor
    lead = None
    for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = box
        if int(c) not in LEAD_CLASSES or y2 <= cy + 3:
            continue
        Z = focal * CAM_H / (y2 - cy)
        X = ((x1 + x2) / 2 - cx) * Z / focal
        if abs(X) < LANE_HALF and (lead is None or Z < lead[0]):
            lead = (Z, x1, y2, x2)

    # Lane corridor shading (from 3 m to the lead car, or to 30 m)
    far = lead[0] if lead else 30
    col = dist_color(lead[0]) if lead else (255, 200, 0)
    poly = np.array([ground(3, -LANE_HALF), ground(3, LANE_HALF),
                     ground(far, LANE_HALF), ground(far, -LANE_HALF)])
    shade = frame.copy()
    cv2.fillPoly(shade, [poly], col)
    frame = cv2.addWeighted(shade, 0.25, frame, 0.75, 0)

    # Distance ticks painted on the road
    for t in TICKS:
        a, b = ground(t, -LANE_HALF), ground(t, LANE_HALF)
        cv2.line(frame, a, b, (255, 255, 255), 1)
        cv2.putText(frame, f"{t}m", (a[0] - 38, a[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Lead car marker at its base
    if lead:
        Z, x1, y2, x2 = lead
        cv2.line(frame, (int(x1), int(y2)), (int(x2), int(y2)), dist_color(Z), 4)
        cv2.putText(frame, f"{Z:.1f} m", (int(x1), min(int(y2) + 22, h - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color(Z), 2)

    hud = f"{len(r.boxes)} objects | {ms:.0f} ms ({1000/ms:.0f} FPS) | " \
          f"{'CPU' if device == 'cpu' else 'GPU'}{' FP16' if half else ' FP32'}"
    cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    vw.write(frame)

vw.release()
print("saved", out_name)