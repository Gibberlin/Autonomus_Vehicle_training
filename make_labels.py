import json, glob, os
import numpy as np
import yaml
from ultralytics import YOLO

W, H, FOV = 640, 480, 90
MIN_VISIBLE, MIN_H = 0.5, 12          # keep boxes >=50% inside frame and >=12 px tall
CLS = {"pedestrian": 0, "vehicle": 2}  # COCO ids: person=0, car=2

f = W / (2 * np.tan(np.radians(FOV / 2)))
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

os.makedirs("dataset/labels", exist_ok=True)
os.makedirs("dataset/preview", exist_ok=True)
import cv2

kept = dropped = 0
for n, mf in enumerate(sorted(glob.glob("dataset/meta/*.json"))):
    meta = json.load(open(mf))
    w2c = np.array(meta["world_to_cam"])
    img = cv2.imread(f"dataset/images/{meta['image']}")
    lines = []
    for o in meta["objects"]:
        v = np.c_[np.array(o["verts"]), np.ones(8)].T          # 4x8
        c = w2c @ v                                             # UE4 camera coords
        std = np.vstack([c[1], -c[2], c[0]])                    # x right, y down, z forward
        if (std[2] <= 0.1).any():                               # any corner behind camera
            dropped += 1; continue
        uv = K @ std
        u, vv = uv[0] / uv[2], uv[1] / uv[2]
        x1, x2, y1, y2 = u.min(), u.max(), vv.min(), vv.max()
        full = (x2 - x1) * (y2 - y1)
        cx1, cx2, cy1, cy2 = max(x1, 0), min(x2, W), max(y1, 0), min(y2, H)
        if cx2 <= cx1 or cy2 <= cy1:
            dropped += 1; continue
        vis = ((cx2 - cx1) * (cy2 - cy1)) / full
        if vis < MIN_VISIBLE or (cy2 - cy1) < MIN_H:
            dropped += 1; continue
        lines.append(f"{CLS[o['cls']]} {(cx1+cx2)/2/W:.6f} {(cy1+cy2)/2/H:.6f} "
                     f"{(cx2-cx1)/W:.6f} {(cy2-cy1)/H:.6f}")
        kept += 1
        cv2.rectangle(img, (int(cx1), int(cy1)), (int(cx2), int(cy2)),
                      (0, 255, 0) if o["cls"] == "vehicle" else (0, 0, 255), 1)
    open(f"dataset/labels/{meta['image'][:-4]}.txt", "w").write("\n".join(lines))
    if n % 40 == 0:
        cv2.imwrite(f"dataset/preview/{meta['image']}", img)

# Dataset config using the model's own COCO class names
names = YOLO("yolov8n.pt").names
yaml.safe_dump({"path": os.path.abspath("dataset"), "train": "images", "val": "images",
                "names": {int(k): v for k, v in names.items()}}, open("dataset.yaml", "w"))
print(f"labels kept: {kept}, dropped: {dropped}")